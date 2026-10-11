// Exports causal-scoring captures for MTP self-distillation: for each input text, the tokens,
// the final-norm hidden state of every predictor and the model's argmax at every predictor.
//
// Input: JSONL, one {"id": "...", "text": "..."} per line. The text is tokenized as raw text
// (no chat template is added; chat markup in the text is tokenized as written).
// Output directory:
//   tokens.i32   int32, every text's tokens concatenated
//   hidden.bf16  BF16 bits, [rows, hidden_size] row-major; row j of a text is predictor j,
//                the state after tokens[0..j] (rows = tokens - 1)
//   argmax.i32   int32, one per row: argmax_v p(v | tokens[0..j])
//   logprob.f32  float, one per row: log p(tokens[j + 1] | tokens[0..j])
//   index.jsonl  {"id", "tokens", "token_offset", "row_offset"} per text, plus a header line

#include "ninfer/engine.h"

#include <nlohmann/json.hpp>

#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using json = nlohmann::json;

constexpr const char* kUsage =
    "usage: ninfer-score-export <artifact.ninfer> --input <texts.jsonl> --output <directory>\n"
    "       [--max-context N] [--kv-dtype bf16|int8|rk4v4-e8] [--device N]\n";

struct Options {
    std::filesystem::path artifact;
    std::filesystem::path input;
    std::filesystem::path output;
    std::uint32_t max_context = 8192;
    ninfer::KvCacheStorage kv = ninfer::KvCacheStorage::RK4V4E8;
    int device                = 0;
};

template <class T> T parse_integer(std::string_view text, const char* name) {
    T value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw std::invalid_argument(std::string("invalid ") + name + ": " + std::string(text));
    }
    return value;
}

Options parse(int argc, char** argv) {
    if (argc < 2) { throw std::invalid_argument(kUsage); }
    Options out;
    out.artifact = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string_view option = argv[i];
        if (i + 1 >= argc) { throw std::invalid_argument(std::string(option) + " needs a value"); }
        const std::string_view value = argv[++i];
        if (option == "--input") {
            out.input = std::string(value);
        } else if (option == "--output") {
            out.output = std::string(value);
        } else if (option == "--max-context") {
            out.max_context = parse_integer<std::uint32_t>(value, "max context");
        } else if (option == "--device") {
            out.device = parse_integer<int>(value, "device");
        } else if (option == "--kv-dtype") {
            if (value == "bf16") {
                out.kv = ninfer::KvCacheStorage::BFloat16;
            } else if (value == "int8") {
                out.kv = ninfer::KvCacheStorage::Int8Group64;
            } else if (value == "rk4v4-e8") {
                out.kv = ninfer::KvCacheStorage::RK4V4E8;
            } else {
                throw std::invalid_argument("unsupported --kv-dtype " + std::string(value));
            }
        } else {
            throw std::invalid_argument("unknown option " + std::string(option) + "\n" + kUsage);
        }
    }
    if (out.input.empty() || out.output.empty()) { throw std::invalid_argument(kUsage); }
    return out;
}

template <class T> void append(std::ofstream& file, const std::vector<T>& values) {
    file.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!file) { throw std::runtime_error("write failed"); }
}

int run(const Options& options) {
    ninfer::EngineOptions engine_options;
    engine_options.artifact_path = options.artifact;
    engine_options.purpose       = ninfer::EnginePurpose::CausalScoring;
    engine_options.device        = options.device;
    engine_options.max_context   = options.max_context;
    engine_options.kv_cache      = options.kv;
    ninfer::Engine engine(std::move(engine_options));

    std::filesystem::create_directories(options.output);
    const auto open = [&](const char* name) {
        std::ofstream file(options.output / name, std::ios::binary | std::ios::trunc);
        if (!file) { throw std::runtime_error(std::string("cannot create ") + name); }
        return file;
    };
    std::ofstream tokens_file  = open("tokens.i32");
    std::ofstream hidden_file  = open("hidden.bf16");
    std::ofstream argmax_file  = open("argmax.i32");
    std::ofstream logprob_file = open("logprob.f32");
    std::ofstream index_file(options.output / "index.jsonl", std::ios::trunc);

    std::ifstream input(options.input);
    if (!input) { throw std::runtime_error("cannot open " + options.input.string()); }

    std::uint64_t token_offset = 0;
    std::uint64_t row_offset   = 0;
    std::uint32_t hidden_size  = 0;
    std::size_t texts = 0, skipped = 0;
    const auto started = std::chrono::steady_clock::now();
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) { continue; }
        const json record = json::parse(line);
        const std::string id   = record.at("id").get<std::string>();
        std::vector<ninfer::TokenId> tokens =
            engine.tokenize_text(record.at("text").get<std::string>());
        if (tokens.size() > options.max_context) { tokens.resize(options.max_context); }
        if (tokens.size() < 2) {
            ++skipped;
            continue;
        }
        const ninfer::CausalScore score =
            engine.score_tokens(tokens, 1, {.hidden = true, .argmax = true});
        if (hidden_size == 0) {
            hidden_size = score.hidden_size;
            index_file << json{{"format", "ninfer-score-export-v1"},
                               {"artifact", options.artifact.string()},
                               {"hidden_size", hidden_size}}
                              .dump()
                       << '\n';
        }
        append(tokens_file, tokens);
        append(hidden_file, score.hidden);
        append(argmax_file, score.argmax);
        append(logprob_file, score.logprobs);
        index_file << json{{"id", id},
                           {"tokens", tokens.size()},
                           {"token_offset", token_offset},
                           {"row_offset", row_offset}}
                          .dump()
                   << '\n';
        token_offset += tokens.size();
        row_offset += score.logprobs.size();
        ++texts;
        if (texts % 25 == 0) {
            const double seconds = std::chrono::duration<double>(
                                       std::chrono::steady_clock::now() - started)
                                       .count();
            std::cerr << "exported " << texts << " texts, " << token_offset << " tokens, "
                      << static_cast<std::uint64_t>(token_offset / seconds) << " tok/s\n";
        }
    }
    std::cerr << "done: " << texts << " texts (" << skipped << " skipped), " << token_offset
              << " tokens, " << row_offset << " rows, hidden " << hidden_size << "\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(parse(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "ninfer-score-export: " << error.what() << '\n';
        return 1;
    }
}

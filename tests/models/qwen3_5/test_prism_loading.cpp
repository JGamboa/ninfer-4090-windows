// Plans a Python-written Bonsai (Prism t2) artifact; tests/models/qwen3_5/prism_loading.py
// writes the valid artifact and the invalid variants and passes the expected refusal.
#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/formats.h"
#include "artifact/reader.h"
#include "models/qwen3_5/load.h"

#include <iostream>
#include <string>
#include <string_view>

namespace {

using namespace ninfer;
using namespace ninfer::models;
namespace qwen = ninfer::models::qwen3_5;
using ninfer::test::artifact_fixture::require;

std::string format_of(const qwen::LoadPlan& plan, const artifact::Reader& reader,
                      qwen::WeightId id) {
    const auto& part = plan.parameter(id).binding.parts.at(0);
    return std::get<artifact::TensorObject>(reader.directory().object(part.object)).format;
}

void valid(const std::filesystem::path& path) {
    artifact::Reader reader(path);
    const auto plan = qwen::plan_load(reader, {.speculative = SpeculativeBackend::Mtp});
    const auto& prism = plan.config().text.prism_hadamard;
    require(prism && prism->rotated_inputs.size() == 14 && prism->signs.size() == 2,
            "prism_hadamard block was not parsed");
    const auto& signs = plan.weights().text.hadamard_signs;
    require(signs.size() == 2 && plan.parameter(signs.at(1024)).shape == artifact::Shape{1024} &&
                plan.parameter(signs.at(2048)).shape == artifact::Shape{2048},
            "Hadamard sign vectors were not bound by width");
    const auto& gdn = std::get<qwen::GdnWeights>(plan.weights().text.layers[0].mixer);
    const auto& attention = std::get<qwen::AttentionWeights>(plan.weights().text.layers[3].mixer);
    const auto& mlp       = std::get<qwen::DenseWeights>(plan.weights().text.layers[3].ffn);
    for (const auto id : {gdn.query, gdn.z, gdn.output, attention.gate, attention.output, mlp.down}) {
        require(format_of(plan, reader, id) == "t2_g128_fp16", "rotated projection is not t2");
    }
    require(format_of(plan, reader, plan.weights().text.output_head) == "t2_g128_fp16" &&
                format_of(plan, reader, gdn.a_projection) == "bf16" &&
                format_of(plan, reader, plan.weights().text.token_embedding) == "q8_g32_fp16",
            "rotated head or unrotated weights changed format");
    require(plan.weights().mtp.has_value(), "copied MTP head was not bound");
}

void rejected(const std::filesystem::path& path, std::string_view expected) {
    try {
        artifact::Reader reader(path);
        (void)qwen::plan_load(reader);
    } catch (const std::exception& error) {
        if (std::string_view(error.what()).find(expected) == std::string_view::npos) {
            std::cerr << "unexpected refusal: " << error.what() << '\n';
            std::exit(1);
        }
        return;
    }
    std::cerr << "invalid Prism artifact was accepted: " << path << '\n';
    std::exit(1);
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 1) { return 77; } // driven by prism_loading.py
        if (argc == 2) {
            valid(argv[1]);
        } else if (argc == 4 && std::string_view(argv[1]) == "--reject") {
            rejected(argv[3], argv[2]);
        } else {
            std::cerr << "usage: " << argv[0] << " ARTIFACT | --reject MESSAGE ARTIFACT\n";
            return 2;
        }
        std::cout << "prism loading checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

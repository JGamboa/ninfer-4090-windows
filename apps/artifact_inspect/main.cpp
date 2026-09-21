// Stand-alone validation tool for the v3 artifact reader/binder work (see
// docs/artifact-v3-port-notes.md). Opens a .ninfer artifact (v2 or v3) and resolves a
// hard-coded set of names representative of what src/targets/qwen3_6_27b's
// bind_groupwise_text_layers() needs, printing what it finds. This exists to validate the
// reader/binder layer against the real v3 artifact without needing a CUDA device or the
// full engine (Binder tracks handles only; nothing here touches the GPU).
#include "artifact/reader.h"

#include <array>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

using ninfer::artifact::ObjectDescriptor;
using ninfer::artifact::Reader;
using ninfer::artifact::TensorDescriptor;

namespace {

void print_descriptor(std::string_view label, const ObjectDescriptor* descriptor) {
    if (descriptor == nullptr) {
        std::printf("  [MISSING] %.*s\n", static_cast<int>(label.size()), label.data());
        return;
    }
    if (const auto* tensor = std::get_if<TensorDescriptor>(descriptor)) {
        std::string shape = "[";
        for (std::size_t i = 0; i < tensor->shape.size(); ++i) {
            if (i != 0) { shape += ","; }
            shape += std::to_string(tensor->shape[i]);
        }
        shape += "]";
        std::printf("  [OK] %.*s -> shape=%s bytes=%llu\n", static_cast<int>(label.size()),
                   label.data(), shape.c_str(),
                   static_cast<unsigned long long>(tensor->bytes));
    } else {
        std::printf("  [OK] %.*s -> resource\n", static_cast<int>(label.size()), label.data());
    }
}

int inspect(const std::filesystem::path& path) {
    std::printf("=== %s ===\n", path.string().c_str());
    Reader reader(path);
    std::printf("model_id=%s weights_id=%s objects=%zu\n", reader.identity().model_id.c_str(),
               reader.identity().weights_id.c_str(), reader.objects().size());

    print_descriptor("text/token_embedding", reader.find("text/token_embedding"));
    print_descriptor("text/final_norm", reader.find("text/final_norm"));
    print_descriptor("frontend/tokenizer.json", reader.find("frontend/tokenizer.json"));

    const auto try_fused = [&](std::string_view label, std::vector<std::string_view> names) {
        try {
            print_descriptor(label, reader.find_fused(names));
        } catch (const std::exception& error) {
            std::printf("  [FAIL] %.*s -> %s\n", static_cast<int>(label.size()), label.data(),
                       error.what());
        }
    };

    // Layer 0 is a linear_attention (GDN) layer; layer 3 is a full_attention layer (see
    // is_full_layer() in bindings.cpp).
    try_fused("text/layers/0/gdn/query_key", {"text/layers/0/gdn/query", "text/layers/0/gdn/key"});
    try_fused("text/layers/0/gdn/value_z", {"text/layers/0/gdn/value", "text/layers/0/gdn/z"});
    try_fused("text/layers/0/mlp/gate_up", {"text/layers/0/mlp/gate", "text/layers/0/mlp/up"});
    try_fused("text/layers/3/attention/query_key",
             {"text/layers/3/attention/query", "text/layers/3/attention/key"});
    try_fused("text/layers/3/attention/gate_value",
             {"text/layers/3/attention/gate", "text/layers/3/attention/value"});
    try_fused("text/layers/3/mlp/gate_up", {"text/layers/3/mlp/gate", "text/layers/3/mlp/up"});
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <artifact.ninfer> [more...]\n", argv[0]);
        return 1;
    }
    int status = 0;
    for (int i = 1; i < argc; ++i) {
        try {
            status |= inspect(argv[i]);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "error opening %s: %s\n", argv[i], error.what());
            status = 1;
        }
    }
    return status;
}

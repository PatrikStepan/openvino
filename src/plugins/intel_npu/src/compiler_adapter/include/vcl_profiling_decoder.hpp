// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "intel_npu/common/iprofiling_decoder.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "openvino/runtime/profiling_info.hpp"

namespace intel_npu {

/**
 * @brief Decodes profiling output through the VCL profiling entry points.
 *
 * Built from a function table alone: decoding needs no compiler instance, so this never calls
 * vclCompilerCreate. Each decode creates and destroys its own profiling handle, so the decoder
 * itself holds no VCL state and is safe to share between graphs.
 */
class VCLProfilingDecoder final : public IProfilingDecoder {
public:
    /**
     * @param functions A populated VCL function table. Holding it keeps the library it was resolved
     *        from loaded, so the decoder cannot outlive the code it dispatches into.
     */
    explicit VCLProfilingDecoder(std::shared_ptr<const VCLFunctionTable> functions);

    std::vector<ov::ProfilingInfo> decode(const std::vector<uint8_t>& profData,
                                          const std::vector<uint8_t>& network) const override;

private:
    std::shared_ptr<const VCLFunctionTable> _functions;
    Logger _logger;
};

/**
 * @brief Loads the VCL compiler library and returns a profiling decoder backed by it.
 *
 * Kept next to makeVCLCompiler() so that the knowledge of how the library is obtained stays in one
 * place. The returned decoder holds the function table as an aliasing pointer, which keeps the
 * library alive without a separate pairing.
 *
 * @note Loading is the expensive part - constructing the decoder itself is just storing a pointer -
 *       so callers that may not need profiling at all should decide before calling this.
 */
std::shared_ptr<IProfilingDecoder> makeVCLProfilingDecoder();

}  // namespace intel_npu

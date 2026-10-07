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
 * @brief Builds a profiling decoder over the given VCL entry points.
 *
 * @note Obtaining the entry points is the expensive part, because it may load the library;
 *       constructing the decoder is just storing a pointer. Callers that may not need profiling at
 *       all should therefore decide before resolving a table to pass here.
 */
std::shared_ptr<IProfilingDecoder> makeVCLProfilingDecoder(std::shared_ptr<const VCLFunctionTable> functions);

}  // namespace intel_npu

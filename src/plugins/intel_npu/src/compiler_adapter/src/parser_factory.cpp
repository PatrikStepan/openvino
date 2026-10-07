// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/parser_factory.hpp"

#include "parser.hpp"
#include "vcl_profiling_decoder.hpp"

namespace intel_npu {

std::unique_ptr<IParser> ParserFactory::getParser(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStructs,
                                                  const bool withProfilingDecoder,
                                                  const VCLFunctionTableProvider& vclFunctions) const {
    OPENVINO_ASSERT(
        zeroInitStructs != nullptr,
        "Could not find an NPU device. The driver compiler requires a valid device to be present in the system.");

    std::shared_ptr<IProfilingDecoder> profilingDecoder;
    if (withProfilingDecoder) {
        OPENVINO_ASSERT(vclFunctions != nullptr,
                        "A profiling decoder was requested but no way to obtain the VCL entry points was given");
        profilingDecoder = makeVCLProfilingDecoder(vclFunctions());
    }

    return std::make_unique<Parser>(zeroInitStructs, std::move(profilingDecoder));
}

}  // namespace intel_npu

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/iparser.hpp"
#include "intel_npu/common/iprofiling_decoder.hpp"
#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace intel_npu {

class ParserFactory final {
public:
    /**
     * @param withProfilingDecoder When true, the parser hands every graph it builds a decoder
     *        resolved through `vclFunctions`, which loads the compiler library. Pass false whenever
     *        profiling is off or the resolved compiler type decodes through the driver instead:
     *        import must keep working on a system with no compiler library present.
     * @param vclFunctions How to obtain the VCL entry points. Invoked only when
     *        `withProfilingDecoder` is true, so it may be empty otherwise.
     */
    std::unique_ptr<IParser> getParser(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStructs,
                                       bool withProfilingDecoder = false,
                                       const VCLFunctionTableProvider& vclFunctions = nullptr) const;
};

}  // namespace intel_npu

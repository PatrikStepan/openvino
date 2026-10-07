// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/iparser.hpp"
#include "intel_npu/common/iprofiling_decoder.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace intel_npu {

class ParserFactory final {
public:
    /**
     * @param profilingDecoder Handed to every graph the parser builds. May be null, in which case
     *        imported graphs cannot decode profiling data - which is the right answer whenever
     *        profiling is off, or the resolved compiler type decodes through the driver instead.
     */
    std::unique_ptr<IParser> getParser(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStructs,
                                       std::shared_ptr<IProfilingDecoder> profilingDecoder = nullptr) const;
};

}  // namespace intel_npu

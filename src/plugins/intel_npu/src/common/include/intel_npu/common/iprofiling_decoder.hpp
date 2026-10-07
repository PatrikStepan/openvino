// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <vector>

#include "openvino/runtime/profiling_info.hpp"

namespace intel_npu {

/**
 * @brief Decodes the raw profiling buffer a graph produced into per-layer profiling info.
 *
 * Deliberately expressed only in OpenVINO and standard types - no VCL, no Level Zero - so that a
 * graph can delegate profiling decode without depending on whichever compiler produced it.
 */
class IProfilingDecoder {
public:
    virtual ~IProfilingDecoder() = default;

    /**
     * @param profData The raw profiling output read back from the device.
     * @param network The compiled blob the profiling data belongs to.
     */
    virtual std::vector<ov::ProfilingInfo> decode(const std::vector<uint8_t>& profData,
                                                  const std::vector<uint8_t>& network) const = 0;
};

}  // namespace intel_npu

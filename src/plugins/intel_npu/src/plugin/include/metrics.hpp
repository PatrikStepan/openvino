// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include "backends.hpp"
#include "intel_npu/common/npu.hpp"
#include "openvino/runtime/properties.hpp"

namespace intel_npu {

class Metrics final {
public:
    Metrics(const std::shared_ptr<const NPUBackends>& backends);

    std::vector<std::string> GetAvailableDevicesNames() const;
    std::string GetFullDeviceName(const std::string& specifiedDeviceName) const;
    IDevice::Uuid GetDeviceUuid(const std::string& specifiedDeviceName) const;
    ov::device::LUID GetDeviceLUID(const std::string& specifiedDeviceName) const;
    std::string GetDeviceArchitecture(const std::string& specifiedDeviceName) const;
    std::string GetBackendName() const;
    uint64_t GetDeviceAllocMemSize(const std::string& specifiedDeviceName) const;
    uint64_t GetDeviceTotalMemSize(const std::string& specifiedDeviceName) const;
    uint32_t GetDriverVersion() const;
    uint32_t GetGraphExtVersion() const;
    uint32_t GetSteppingNumber(const std::string& specifiedDeviceName) const;
    uint32_t GetMaxTiles(const std::string& specifiedDeviceName) const;
    ov::device::PCIInfo GetPciInfo(const std::string& specifiedDeviceName) const;
    std::map<ov::element::Type, float> GetGops(const std::string& specifiedDeviceName) const;
    ov::device::Type GetDeviceType(const std::string& specifiedDeviceName) const;
    std::string GetCompilationPlatform(const std::string_view platform, const std::string& deviceId) const;
    bool IsCommandQueueExtSupported() const;

    ~Metrics() = default;

private:
    const std::shared_ptr<const NPUBackends> _backends;

    std::string getDeviceName(const std::string& specifiedDeviceName) const;
};

}  // namespace intel_npu

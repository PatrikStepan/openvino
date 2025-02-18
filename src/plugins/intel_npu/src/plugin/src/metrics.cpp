// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Plugin
#include "metrics.hpp"

#include "intel_npu/common/device_helpers.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace intel_npu {

Metrics::Metrics(const std::shared_ptr<const NPUBackends>& backends) : _backends(backends) {
}

std::vector<std::string> Metrics::GetAvailableDevicesNames() const {
    return _backends == nullptr ? std::vector<std::string>() : _backends->getAvailableDevicesNames();
}

std::string Metrics::GetFullDeviceName(const std::string& specifiedDeviceName) const {
    const auto devName = getDeviceName(specifiedDeviceName);
    auto device = _backends->getDevice(devName);
    if (device) {
        return device->getFullDeviceName();
    }
    OPENVINO_THROW("No device with name '", specifiedDeviceName, "' is available");
}

std::string Metrics::GetDeviceArchitecture(const std::string& specifiedDeviceName) const {
    const auto devName = getDeviceName(specifiedDeviceName);
    return utils::getPlatformByDeviceName(devName);
}

IDevice::Uuid Metrics::GetDeviceUuid(const std::string& specifiedDeviceName) const {
    const auto devName = getDeviceName(specifiedDeviceName);
    auto device = _backends->getDevice(devName);
    if (device) {
        return device->getUuid();
    }
    return IDevice::Uuid{};
}

ov::device::LUID Metrics::GetDeviceLUID(const std::string& specifiedDeviceName) const {
    const auto devName = getDeviceName(specifiedDeviceName);
    auto device = _backends->getDevice(devName);
    if (device) {
        return device->getLUID();
    }
    return ov::device::LUID{{
        0,
    }};
}

std::string Metrics::GetBackendName() const {
    if (_backends == nullptr) {
        OPENVINO_THROW("No available backends");
    }

    return _backends->getBackendName();
}

uint32_t Metrics::GetDriverVersion() const {
    if (_backends == nullptr) {
        OPENVINO_THROW("No available backends");
    }

    return _backends->getDriverVersion();
}

uint32_t Metrics::GetGraphExtVersion() const {
    if (_backends == nullptr) {
        OPENVINO_THROW("No available backends");
    }

    return _backends->getGraphExtVersion();
}

uint32_t Metrics::GetSteppingNumber(const std::string& specifiedDeviceName) const {
    const auto devName = getDeviceName(specifiedDeviceName);
    auto device = _backends->getDevice(devName);
    if (device) {
        return device->getSubDevId();
    }
    OPENVINO_THROW("No device with name '", specifiedDeviceName, "' is available");
}

uint32_t Metrics::GetMaxTiles(const std::string& specifiedDeviceName) const {
    const auto devName = getDeviceName(specifiedDeviceName);
    auto device = _backends->getDevice(devName);
    if (device) {
        return device->getMaxNumSlices();
    }
    OPENVINO_THROW("No device with name '", specifiedDeviceName, "' is available");
}

uint64_t Metrics::GetDeviceAllocMemSize(const std::string& specifiedDeviceName) const {
    const auto devName = getDeviceName(specifiedDeviceName);
    auto device = _backends->getDevice(devName);
    if (device) {
        return device->getAllocMemSize();
    }
    OPENVINO_THROW("No device with name '", specifiedDeviceName, "' is available");
}

uint64_t Metrics::GetDeviceTotalMemSize(const std::string& specifiedDeviceName) const {
    const auto devName = getDeviceName(specifiedDeviceName);
    auto device = _backends->getDevice(devName);
    if (device) {
        return device->getTotalMemSize();
    }
    OPENVINO_THROW("No device with name '", specifiedDeviceName, "' is available");
}

std::string Metrics::getDeviceName(const std::string& specifiedDeviceName) const {
    std::vector<std::string> devNames;
    if (_backends == nullptr || (devNames = _backends->getAvailableDevicesNames()).empty()) {
        OPENVINO_THROW("No available devices");
    }

    // In case of single device and empty input from user we should use the first element from the device list
    if (specifiedDeviceName.empty()) {
        if (devNames.size() == 1) {
            return devNames[0];
        } else {
            OPENVINO_THROW("The device name was not specified. Please specify device name by providing DEVICE_ID");
        }
    }

    return specifiedDeviceName;
}

ov::device::PCIInfo Metrics::GetPciInfo(const std::string& specifiedDeviceName) const {
    const auto devName = getDeviceName(specifiedDeviceName);
    auto device = _backends->getDevice(devName);
    if (device != nullptr) {
        return device->getPciInfo();
    }
    OPENVINO_THROW("No device with name '", specifiedDeviceName, "' is available");
}

std::map<ov::element::Type, float> Metrics::GetGops(const std::string& specifiedDeviceName) const {
    const auto devName = getDeviceName(specifiedDeviceName);
    auto device = _backends->getDevice(devName);
    if (device != nullptr) {
        return device->getGops();
    }
    OPENVINO_THROW("No device with name '", specifiedDeviceName, "' is available");
}

ov::device::Type Metrics::GetDeviceType(const std::string& specifiedDeviceName) const {
    const auto devName = getDeviceName(specifiedDeviceName);
    auto device = _backends->getDevice(devName);
    if (device != nullptr) {
        return device->getDeviceType();
    }
    OPENVINO_THROW("No device with name '", specifiedDeviceName, "' is available");
}

std::string Metrics::GetCompilationPlatform(const std::string_view platform, const std::string& deviceId) const {
    return _backends->getCompilationPlatform(platform, deviceId);
}

bool Metrics::IsCommandQueueExtSupported() const {
    return _backends->isCommandQueueExtSupported();
}

}  // namespace intel_npu

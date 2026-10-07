// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "openvino/core/except.hpp"

namespace intel_npu {

/**
 * @brief Fetches the last error VCL recorded, for appending to a diagnostic.
 *
 * Never throws and never propagates a VCL failure: it is only ever called while building an error
 * message, so a failure here must not replace the error being reported. Lifted out of
 * compiler_impl.cpp so that the profiling decoder can report VCL failures the same way without
 * depending on the compiler.
 *
 * @param functions The table to dispatch through, passed explicitly so this is usable outside a
 *        class that holds one.
 */
inline std::string getLatestVCLLog(const VCLFunctionTable& functions, vcl_log_handle_t logHandle) {
    Logger logger("VCLAPI", Logger::global().level());
    logger.debug("getLatestVCLLog start");

    vcl_version_info_t compilerVersion;
    vcl_version_info_t profilingVersion;
    vcl_result_t ret = functions.vclGetVersion(&compilerVersion, &profilingVersion);

    if (ret != VCL_RESULT_SUCCESS || compilerVersion.major < 3) {
        logger.warning("Failed to get VCL version: 0x%x", ret);
        return "Can not get VCL log, VCL version is too old!";
    }

    // Get log size. A null handle yields the global error log.
    size_t size = 0;
    ret = functions.vclLogHandleGetString(logHandle, &size, nullptr);
    if (VCL_RESULT_SUCCESS != ret) {
        return "Failed to get size of latest VCL log";
    }

    if (size == 0) {
        return "No error stored in VCL when error detected";
    }

    std::string logContent{};
    logContent.resize(size);
    ret = functions.vclLogHandleGetString(logHandle, &size, logContent.data());
    if (VCL_RESULT_SUCCESS != ret) {
        return "Size of latest error log > 0, failed to get content";
    }
    logger.debug("getLatestVCLLog end");
    return logContent;
}

}  // namespace intel_npu

/**
 * @brief Throws with the VCL error log appended when `ret` is not VCL_RESULT_SUCCESS.
 * @param functions The function table to fetch the error log through, passed explicitly rather than
 * captured from the enclosing scope so the macro is usable outside a member function.
 */
#define THROW_ON_FAIL_FOR_VCL(functions, step, ret, logHandle)                            \
    do {                                                                                  \
        const vcl_result_t vclResult_ = (ret);                                            \
        if (vclResult_ != VCL_RESULT_SUCCESS) {                                           \
            OPENVINO_THROW("Failed to call VCL API : ",                                   \
                           step,                                                          \
                           " result: 0x",                                                 \
                           std::hex,                                                      \
                           vclResult_,                                                    \
                           " - ",                                                         \
                           ::intel_npu::getLatestVCLLog((functions), logHandle));         \
        }                                                                                 \
    } while (0)

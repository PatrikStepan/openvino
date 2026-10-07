// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vcl_profiling_decoder.hpp"

#include <gtest/gtest.h>
#include <level_zero/ze_api.h>
#include <ze_graph_ext.h>
#include <ze_graph_profiling_ext.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "fake_vcl.hpp"
#include "openvino/core/except.hpp"

using ::fake_vcl::FakeVcl;
using ::intel_npu::VCLProfilingDecoder;

namespace {

// Decoding needs the function table and nothing else - no compiler, no driver, no NPU. The whole
// point of VCLProfilingDecoder is that these cases never call vclCompilerCreate, which is asserted
// explicitly below.
class VCLProfilingDecoderTest : public ::testing::Test {
protected:
    FakeVcl fake;

    std::shared_ptr<VCLProfilingDecoder> makeDecoder() {
        return std::make_shared<VCLProfilingDecoder>(fake.functions());
    }
};

TEST_F(VCLProfilingDecoderTest, ConstructionCreatesNoCompiler) {
    auto decoder = makeDecoder();
    EXPECT_EQ(fake.callCount("vclCompilerCreate"), 0u);
}

TEST_F(VCLProfilingDecoderTest, DecodeCreatesNoCompiler) {
    fake.profilingPayload.assign(sizeof(ze_profiling_layer_info), 0);
    auto decoder = makeDecoder();

    (void)decoder->decode({1}, {2});

    // The regression this class exists to prevent: a compiler per profiling fetch.
    EXPECT_EQ(fake.callCount("vclCompilerCreate"), 0u);
    EXPECT_EQ(fake.callCount("vclCompilerGetProperties"), 0u);
}

TEST_F(VCLProfilingDecoderTest, DecodeFollowsCreateGetDestroyOrdering) {
    fake.profilingPayload.assign(2 * sizeof(ze_profiling_layer_info), 0);
    auto decoder = makeDecoder();

    const std::vector<uint8_t> profData{1, 2, 3};
    const std::vector<uint8_t> network{4, 5, 6, 7};
    const auto info = decoder->decode(profData, network);

    EXPECT_EQ(fake.callCount("vclProfilingCreate"), 1u);
    EXPECT_EQ(fake.callCount("vclProfilingGetProperties"), 1u);
    EXPECT_EQ(fake.callCount("vclGetDecodedProfilingBuffer"), 1u);
    EXPECT_EQ(fake.profilingDestroyCount, 1);

    EXPECT_LT(fake.indexOf("vclProfilingCreate"), fake.indexOf("vclProfilingGetProperties"));
    EXPECT_LT(fake.indexOf("vclProfilingGetProperties"), fake.indexOf("vclGetDecodedProfilingBuffer"));
    EXPECT_LT(fake.indexOf("vclGetDecodedProfilingBuffer"), fake.indexOf("vclProfilingDestroy"));

    // One entry per ze_profiling_layer_info in the returned buffer.
    EXPECT_EQ(info.size(), 2u);
}

TEST_F(VCLProfilingDecoderTest, DecodeSizesByLayerInfoStride) {
    fake.profilingPayload.assign(5 * sizeof(ze_profiling_layer_info), 0);
    auto decoder = makeDecoder();
    EXPECT_EQ(decoder->decode({1}, {2}).size(), 5u);
}

TEST_F(VCLProfilingDecoderTest, DecodeThrowsOnNullData) {
    fake.forceNullProfilingData = true;
    auto decoder = makeDecoder();

    try {
        decoder->decode({1}, {2});
        FAIL() << "Expected a throw on NULL profiling data";
    } catch (const ov::Exception& error) {
        EXPECT_NE(std::string(error.what()).find("Failed to get VCL profiling output"), std::string::npos);
    }
}

TEST_F(VCLProfilingDecoderTest, DecodeThrowsWhenCreateFails) {
    // A decodable payload, so the scripted vclProfilingCreate failure is the only reason to throw.
    fake.profilingPayload.assign(sizeof(ze_profiling_layer_info), 0);
    auto decoder = makeDecoder();
    fake.failWith("vclProfilingCreate", VCL_RESULT_ERROR_UNKNOWN);

    try {
        decoder->decode({1}, {2});
        FAIL() << "Expected a throw on vclProfilingCreate failure";
    } catch (const ov::Exception& error) {
        EXPECT_NE(std::string(error.what()).find("vclProfilingCreate"), std::string::npos);
    }
    // Nothing was created, so nothing must be destroyed.
    EXPECT_EQ(fake.profilingDestroyCount, 0);
}

TEST_F(VCLProfilingDecoderTest, DecodeThrowsWhenDestroyFails) {
    // The payload must decode successfully, otherwise the null-data guard throws first and the
    // destroy-failure path is never reached.
    fake.profilingPayload.assign(sizeof(ze_profiling_layer_info), 0);
    auto decoder = makeDecoder();
    fake.failWith("vclProfilingDestroy", VCL_RESULT_ERROR_UNKNOWN);

    try {
        decoder->decode({1}, {2});
        FAIL() << "Expected a throw on vclProfilingDestroy failure";
    } catch (const ov::Exception& error) {
        EXPECT_NE(std::string(error.what()).find("vclProfilingDestroy"), std::string::npos);
    }
    // The decode ran to completion before destroy was attempted.
    EXPECT_EQ(fake.callCount("vclGetDecodedProfilingBuffer"), 1u);
    EXPECT_EQ(fake.profilingDestroyCount, 1);
}

TEST_F(VCLProfilingDecoderTest, DecodeIsRepeatableOnOneDecoder) {
    // Each decode owns its profiling handle, so the decoder holds no state between calls.
    fake.profilingPayload.assign(3 * sizeof(ze_profiling_layer_info), 0);
    auto decoder = makeDecoder();

    EXPECT_EQ(decoder->decode({1}, {2}).size(), 3u);
    EXPECT_EQ(decoder->decode({1}, {2}).size(), 3u);

    EXPECT_EQ(fake.callCount("vclProfilingCreate"), 2u);
    EXPECT_EQ(fake.profilingDestroyCount, 2);
}

TEST_F(VCLProfilingDecoderTest, RejectsNullFunctionTable) {
    EXPECT_THROW((VCLProfilingDecoder{nullptr}), ov::Exception);
}

TEST_F(VCLProfilingDecoderTest, RejectsUnpopulatedFunctionTable) {
    auto empty = std::make_shared<const intel_npu::VCLFunctionTable>();
    EXPECT_THROW((VCLProfilingDecoder{empty}), ov::Exception);
}

}  // namespace

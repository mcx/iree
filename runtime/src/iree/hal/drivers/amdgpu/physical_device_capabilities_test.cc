// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/physical_device_capabilities.h"

#include <array>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

static hsa_agent_t Agent(uint64_t handle) {
  hsa_agent_t agent = {};
  agent.handle = handle;
  return agent;
}

static hsa_amd_memory_pool_t MemoryPool(uint64_t handle) {
  hsa_amd_memory_pool_t memory_pool = {};
  memory_pool.handle = handle;
  return memory_pool;
}

static hsa_amd_hdp_flush_t HdpFlush(uintptr_t mem_flush_control,
                                    uintptr_t register_flush_control) {
  hsa_amd_hdp_flush_t hdp_flush = {};
  hdp_flush.HDP_MEM_FLUSH_CNTL = reinterpret_cast<uint32_t*>(mem_flush_control);
  hdp_flush.HDP_REG_FLUSH_CNTL =
      reinterpret_cast<uint32_t*>(register_flush_control);
  return hdp_flush;
}

static iree_hal_amdgpu_gfxip_version_t GfxIp(uint16_t major, uint16_t minor,
                                             uint16_t stepping) {
  iree_hal_amdgpu_gfxip_version_t version = {};
  version.major = major;
  version.minor = minor;
  version.stepping = stepping;
  return version;
}

class PhysicalDeviceCapabilitiesTest : public ::testing::Test {
 protected:
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t
  MakeCoarseMemorySelection() {
    iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t selection = {};
    selection.device_agent = Agent(10);
    selection.memory_pool = MemoryPool(20);
    selection.gfxip_version = GfxIp(11, 0, 0);
    selection.cpu.agents = cpu_agents_.data();
    selection.cpu.access = cpu_access_.data();
    selection.cpu.count = cpu_agents_.size();
    selection.hdp.registers = HdpFlush(0xCAFE, 0xBEEF);
    selection.flags =
        IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_SELECTION_FLAG_HOST_WRITE_PUBLICATION_SUPPORTED;
    return selection;
  }

  std::array<hsa_agent_t, 2> cpu_agents_ = {Agent(1), Agent(2)};
  std::array<hsa_amd_memory_pool_access_t, 2> cpu_access_ = {
      HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT,
      HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT};
};

TEST_F(PhysicalDeviceCapabilitiesTest, SelectsAvailableCoarseMemory) {
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t selection =
      MakeCoarseMemorySelection();
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_t capability;
  IREE_ASSERT_OK(iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
      &selection, &capability));

  EXPECT_TRUE(iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
      &capability));
  EXPECT_EQ(capability.memory_pool.handle, selection.memory_pool.handle);
  ASSERT_EQ(capability.access_agent_count, 3u);
  EXPECT_EQ(capability.access_agents[0].handle, cpu_agents_[0].handle);
  EXPECT_EQ(capability.access_agents[1].handle, cpu_agents_[1].handle);
  EXPECT_EQ(capability.access_agents[2].handle, selection.device_agent.handle);
  EXPECT_EQ(capability.host_write_publication.mode,
            IREE_HAL_AMDGPU_KERNARG_RING_PUBLICATION_MODE_HDP_FLUSH);
  EXPECT_EQ(capability.host_write_publication.hdp_mem_flush_control,
            selection.hdp.registers.HDP_MEM_FLUSH_CNTL);
  EXPECT_TRUE(iree_all_bits_set(
      capability.flags,
      IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_FLAG_AVAILABLE |
          IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_FLAG_HDP_FLUSH));
}

TEST_F(PhysicalDeviceCapabilitiesTest, EmptyInputsDisableCoarseMemory) {
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t selection =
      MakeCoarseMemorySelection();
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_t capability;

  selection.memory_pool = MemoryPool(0);
  IREE_ASSERT_OK(iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
      &selection, &capability));
  EXPECT_FALSE(iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
      &capability));

  selection = MakeCoarseMemorySelection();
  selection.cpu.count = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
      &selection, &capability));
  EXPECT_FALSE(iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
      &capability));
}

TEST_F(PhysicalDeviceCapabilitiesTest, PublicationGatesDisableCoarseMemory) {
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t selection =
      MakeCoarseMemorySelection();
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_t capability;

  selection.flags =
      IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_SELECTION_FLAG_NONE;
  IREE_ASSERT_OK(iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
      &selection, &capability));
  EXPECT_FALSE(iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
      &capability));

  selection = MakeCoarseMemorySelection();
  selection.hdp.registers.HDP_MEM_FLUSH_CNTL = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
      &selection, &capability));
  EXPECT_FALSE(iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
      &capability));

  selection = MakeCoarseMemorySelection();
  selection.hdp.registers.HDP_REG_FLUSH_CNTL = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
      &selection, &capability));
  EXPECT_FALSE(iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
      &capability));
}

TEST_F(PhysicalDeviceCapabilitiesTest, GfxIpGatesHdpPublication) {
  EXPECT_FALSE(
      iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(GfxIp(9, 0, 7)));
  EXPECT_TRUE(
      iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(GfxIp(9, 0, 8)));
  EXPECT_FALSE(
      iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(GfxIp(10, 0, 0)));
  EXPECT_FALSE(
      iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(GfxIp(10, 1, 0)));
  EXPECT_TRUE(
      iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(GfxIp(10, 3, 0)));
  EXPECT_TRUE(
      iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(GfxIp(11, 0, 0)));
}

TEST_F(PhysicalDeviceCapabilitiesTest, UnsupportedGfxIpDisablesCoarseMemory) {
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t selection =
      MakeCoarseMemorySelection();
  selection.gfxip_version = GfxIp(10, 1, 0);
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_t capability;
  IREE_ASSERT_OK(iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
      &selection, &capability));
  EXPECT_FALSE(iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
      &capability));
}

TEST_F(PhysicalDeviceCapabilitiesTest, CpuAccessGatesCoarseMemory) {
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t selection =
      MakeCoarseMemorySelection();
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_t capability;

  cpu_access_[1] = HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
  IREE_ASSERT_OK(iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
      &selection, &capability));
  EXPECT_FALSE(iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
      &capability));

  cpu_access_[1] = (hsa_amd_memory_pool_access_t)99;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
                            &selection, &capability));
}

TEST_F(PhysicalDeviceCapabilitiesTest, CpuAccessInputsAreRequiredWhenNeeded) {
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t selection =
      MakeCoarseMemorySelection();
  selection.cpu.agents = nullptr;
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_t capability;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
                            &selection, &capability));

  selection = MakeCoarseMemorySelection();
  selection.cpu.access = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
                            &selection, &capability));
}

TEST_F(PhysicalDeviceCapabilitiesTest, TooManyCpuAgentsFails) {
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t selection =
      MakeCoarseMemorySelection();
  selection.cpu.count = IREE_HAL_AMDGPU_MAX_CPU_AGENT + 1;
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_t capability;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
                            &selection, &capability));
}

TEST_F(PhysicalDeviceCapabilitiesTest, SelectsPrepublishedKernargStorage) {
  iree_hal_amdgpu_aql_prepublished_kernarg_storage_t storage =
      iree_hal_amdgpu_select_prepublished_kernarg_storage(MemoryPool(0));
  EXPECT_EQ(storage.strategy,
            IREE_HAL_AMDGPU_AQL_PREPUBLISHED_KERNARG_STORAGE_STRATEGY_DISABLED);

  storage = iree_hal_amdgpu_select_prepublished_kernarg_storage(MemoryPool(42));
  EXPECT_EQ(
      storage.strategy,
      IREE_HAL_AMDGPU_AQL_PREPUBLISHED_KERNARG_STORAGE_STRATEGY_DEVICE_FINE_HOST_COHERENT);
  EXPECT_TRUE(iree_all_bits_set(storage.buffer_params.type,
                                IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                                    IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                                    IREE_HAL_MEMORY_TYPE_HOST_COHERENT));
}

}  // namespace
}  // namespace iree::hal::amdgpu

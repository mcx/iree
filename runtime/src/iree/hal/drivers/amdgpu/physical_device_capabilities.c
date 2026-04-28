// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/physical_device_capabilities.h"

#include <string.h>

bool iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
    const iree_hal_amdgpu_cpu_visible_device_coarse_memory_t* memory) {
  return iree_any_bit_set(
      memory->flags,
      IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_FLAG_AVAILABLE);
}

static bool iree_hal_amdgpu_memory_pool_access_is_valid(
    hsa_amd_memory_pool_access_t access) {
  switch (access) {
    case HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED:
    case HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT:
    case HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT:
      return true;
    default:
      return false;
  }
}

static bool iree_hal_amdgpu_gfxip_is_pre_gfx908(
    iree_hal_amdgpu_gfxip_version_t version) {
  return version.major < 9 ||
         (version.major == 9 && version.minor == 0 && version.stepping < 8);
}

static bool iree_hal_amdgpu_gfxip_is_gfx101x(
    iree_hal_amdgpu_gfxip_version_t version) {
  return version.major == 10 && (version.minor == 0 || version.minor == 1);
}

bool iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(
    iree_hal_amdgpu_gfxip_version_t version) {
  // Matches the HDP workaround eligibility in CLR's setKernelArgImpl. Devices
  // outside this set stay on host kernarg memory unless we add a first-class
  // readback publication mode.
  return !iree_hal_amdgpu_gfxip_is_pre_gfx908(version) &&
         !iree_hal_amdgpu_gfxip_is_gfx101x(version);
}

iree_status_t iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
    const iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t*
        selection,
    iree_hal_amdgpu_cpu_visible_device_coarse_memory_t* out_memory) {
  IREE_ASSERT_ARGUMENT(selection);
  IREE_ASSERT_ARGUMENT(out_memory);
  memset(out_memory, 0, sizeof(*out_memory));

  if (!selection->memory_pool.handle || selection->cpu.count == 0) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(selection->cpu.count > IREE_HAL_AMDGPU_MAX_CPU_AGENT)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU topology has %" PRIhsz
        " CPU agents but CPU-visible coarse memory tracks at most %d",
        selection->cpu.count, IREE_HAL_AMDGPU_MAX_CPU_AGENT);
  }
  if (!iree_any_bit_set(
          selection->flags,
          IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_SELECTION_FLAG_HOST_WRITE_PUBLICATION_SUPPORTED)) {
    return iree_ok_status();
  }
  if (!iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(
          selection->gfxip_version)) {
    return iree_ok_status();
  }
  if (!selection->hdp.registers.HDP_MEM_FLUSH_CNTL ||
      !selection->hdp.registers.HDP_REG_FLUSH_CNTL) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(!selection->cpu.agents || !selection->cpu.access)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "CPU-visible device-coarse memory selection requires CPU agents and "
        "access modes");
  }

  for (iree_host_size_t i = 0; i < selection->cpu.count; ++i) {
    const hsa_amd_memory_pool_access_t access = selection->cpu.access[i];
    if (IREE_UNLIKELY(!iree_hal_amdgpu_memory_pool_access_is_valid(access))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "HSA reported unknown memory pool access mode %u",
                              (uint32_t)access);
    }
    if (access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
      return iree_ok_status();
    }
  }

  iree_host_size_t access_agent_count = 0;
  for (iree_host_size_t i = 0; i < selection->cpu.count; ++i) {
    out_memory->access_agents[access_agent_count++] = selection->cpu.agents[i];
  }
  out_memory->access_agents[access_agent_count++] = selection->device_agent;
  out_memory->memory_pool = selection->memory_pool;
  out_memory->access_agent_count = access_agent_count;
  out_memory->host_write_publication =
      (iree_hal_amdgpu_kernarg_ring_publication_t){
          .mode = IREE_HAL_AMDGPU_KERNARG_RING_PUBLICATION_MODE_HDP_FLUSH,
          .hdp_mem_flush_control = selection->hdp.registers.HDP_MEM_FLUSH_CNTL,
      };
  out_memory->flags =
      IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_FLAG_AVAILABLE |
      IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_FLAG_HDP_FLUSH;
  return iree_ok_status();
}

iree_hal_amdgpu_aql_prepublished_kernarg_storage_t
iree_hal_amdgpu_select_prepublished_kernarg_storage(
    hsa_amd_memory_pool_t fine_block_memory_pool) {
  if (!fine_block_memory_pool.handle) {
    return iree_hal_amdgpu_aql_prepublished_kernarg_storage_disabled();
  }
  return iree_hal_amdgpu_aql_prepublished_kernarg_storage_device_fine_host_coherent();
}

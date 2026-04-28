// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_PHYSICAL_DEVICE_CAPABILITIES_H_
#define IREE_HAL_DRIVERS_AMDGPU_PHYSICAL_DEVICE_CAPABILITIES_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/aql_prepublished_kernarg_storage.h"
#include "iree/hal/drivers/amdgpu/util/kernarg_ring.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/target_id.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum iree_hal_amdgpu_cpu_visible_device_coarse_memory_flag_bits_e {
  IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_FLAG_NONE = 0u,
  // All CPU agents can access the GPU coarse-grained memory pool and the
  // driver knows how to publish CPU writes before GPU consumption.
  IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_FLAG_AVAILABLE = 1u << 0,
  // CPU writes require an HDP flush before the GPU consumes the memory.
  IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_FLAG_HDP_FLUSH = 1u << 1,
} iree_hal_amdgpu_cpu_visible_device_coarse_memory_flag_bits_t;

typedef uint32_t iree_hal_amdgpu_cpu_visible_device_coarse_memory_flags_t;

// Physical-device capability for CPU-visible GPU coarse-grained memory.
typedef struct iree_hal_amdgpu_cpu_visible_device_coarse_memory_t {
  // GPU coarse-grained HSA memory pool CPU agents can access.
  hsa_amd_memory_pool_t memory_pool;
  // Agents granted access for allocations that use |memory_pool|.
  hsa_agent_t access_agents[IREE_HAL_AMDGPU_MAX_CPU_AGENT + 1];
  // Number of valid entries in |access_agents|.
  iree_host_size_t access_agent_count;
  // Publication required after CPU writes and before GPU consumption.
  iree_hal_amdgpu_kernarg_ring_publication_t host_write_publication;
  // Capability flags from
  // iree_hal_amdgpu_cpu_visible_device_coarse_memory_flag_bits_t.
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_flags_t flags;
} iree_hal_amdgpu_cpu_visible_device_coarse_memory_t;

typedef enum iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_flag_bits_e {
  IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_SELECTION_FLAG_NONE = 0u,
  // Host writes can be published for CPU-visible device-coarse memory.
  IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_SELECTION_FLAG_HOST_WRITE_PUBLICATION_SUPPORTED =
      1u << 0,
} iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_flag_bits_t;

typedef uint32_t
    iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_flags_t;

// Queried facts used to select CPU-visible device-coarse memory capability.
typedef struct iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t {
  // GPU agent that owns |memory_pool|.
  hsa_agent_t device_agent;
  // GPU coarse-grained memory pool being considered.
  hsa_amd_memory_pool_t memory_pool;
  // Parsed gfx IP version for HDP publication eligibility.
  iree_hal_amdgpu_gfxip_version_t gfxip_version;
  // CPU agents and their access to |memory_pool|.
  struct {
    // CPU agents that may write the memory.
    const hsa_agent_t* agents;
    // Per-CPU-agent access mode for |memory_pool|.
    const hsa_amd_memory_pool_access_t* access;
    // Number of entries in |agents| and |access|.
    iree_host_size_t count;
  } cpu;
  // HDP publication registers reported by HSA.
  struct {
    // Raw HSA HDP flush register descriptor.
    hsa_amd_hdp_flush_t registers;
  } hdp;
  // Selection flags from
  // iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_flag_bits_t.
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_flags_t flags;
} iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t;

// Returns true if CPU-visible device-coarse memory is available.
bool iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
    const iree_hal_amdgpu_cpu_visible_device_coarse_memory_t* memory);

// Returns true if the gfx IP family permits HDP kernarg publication.
bool iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(
    iree_hal_amdgpu_gfxip_version_t version);

// Selects CPU-visible device-coarse memory from already-queried topology facts.
iree_status_t iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
    const iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t*
        selection,
    iree_hal_amdgpu_cpu_visible_device_coarse_memory_t* out_memory);

// Selects command-buffer prepublished kernarg storage from queried memory
// pools.
iree_hal_amdgpu_aql_prepublished_kernarg_storage_t
iree_hal_amdgpu_select_prepublished_kernarg_storage(
    hsa_amd_memory_pool_t fine_block_memory_pool);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_PHYSICAL_DEVICE_CAPABILITIES_H_

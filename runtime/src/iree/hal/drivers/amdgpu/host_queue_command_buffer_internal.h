// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_INTERNAL_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_INTERNAL_H_

#include "iree/hal/drivers/amdgpu/host_queue_command_buffer.h"
#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef uint32_t iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t;
enum iree_hal_amdgpu_host_queue_command_buffer_packet_flag_bits_t {
  IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_NONE = 0u,
  // Packet must participate in the command-buffer execution dependency chain.
  IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_EXECUTION_BARRIER =
      1u << 0,
  // Packet owns queue completion and releases user-visible signal semaphores.
  IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_FINAL = 1u << 1,
  // First bit of the two-bit acquire fence scope field in packet flags.
  IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_ACQUIRE_SCOPE_SHIFT = 2,
  // Bit mask of the two-bit acquire fence scope field in packet flags.
  IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_ACQUIRE_SCOPE_MASK = 0x0Cu,
  // First bit of the two-bit release fence scope field in packet flags.
  IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_RELEASE_SCOPE_SHIFT = 4,
  // Bit mask of the two-bit release fence scope field in packet flags.
  IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_RELEASE_SCOPE_MASK = 0x30u,
};

// Returns |flags| with its encoded fence scope fields replaced.
static inline iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t
iree_hal_amdgpu_host_queue_command_buffer_packet_flags_set_fence_scopes(
    iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t flags,
    iree_hsa_fence_scope_t acquire_scope,
    iree_hsa_fence_scope_t release_scope) {
  flags &=
      ~(IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_ACQUIRE_SCOPE_MASK |
        IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_RELEASE_SCOPE_MASK);
  flags |=
      ((uint32_t)acquire_scope & 0x3u)
      << IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_ACQUIRE_SCOPE_SHIFT;
  flags |=
      ((uint32_t)release_scope & 0x3u)
      << IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_RELEASE_SCOPE_SHIFT;
  return flags;
}

// Returns one encoded fence scope field from packet flags.
static inline iree_hsa_fence_scope_t
iree_hal_amdgpu_host_queue_command_buffer_packet_flags_fence_scope(
    iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t flags,
    uint32_t mask, uint32_t shift) {
  return (iree_hsa_fence_scope_t)((flags & mask) >> shift);
}

// Returns the acquire fence scope encoded in packet flags.
static inline iree_hsa_fence_scope_t
iree_hal_amdgpu_host_queue_command_buffer_packet_flags_acquire_scope(
    iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t flags) {
  return iree_hal_amdgpu_host_queue_command_buffer_packet_flags_fence_scope(
      flags,
      IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_ACQUIRE_SCOPE_MASK,
      IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_ACQUIRE_SCOPE_SHIFT);
}

// Returns the release fence scope encoded in packet flags.
static inline iree_hsa_fence_scope_t
iree_hal_amdgpu_host_queue_command_buffer_packet_flags_release_scope(
    iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t flags) {
  return iree_hal_amdgpu_host_queue_command_buffer_packet_flags_fence_scope(
      flags,
      IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_RELEASE_SCOPE_MASK,
      IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_RELEASE_SCOPE_SHIFT);
}

// Computes AQL packet control for one replayed command-buffer packet.
iree_hal_amdgpu_aql_packet_control_t
iree_hal_amdgpu_host_queue_command_buffer_packet_control(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    uint32_t packet_index, iree_hsa_fence_scope_t minimum_acquire_scope,
    iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t packet_flags);

// Submits one finalized command-buffer block to the queue.
iree_status_t iree_hal_amdgpu_host_queue_submit_command_buffer_block(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    iree_hal_resource_set_t** inout_binding_resource_set,
    iree_hal_amdgpu_reclaim_action_t pre_signal_action,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready);

// Validates that a metadata-only command-buffer program can be replayed.
iree_status_t iree_hal_amdgpu_host_queue_validate_metadata_commands(
    const iree_hal_amdgpu_aql_program_t* program);

// Starts multi-block command-buffer replay.
iree_status_t iree_hal_amdgpu_command_buffer_replay_start_under_lock(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t execute_flags,
    iree_hal_resource_set_t** inout_binding_resource_set);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_INTERNAL_H_

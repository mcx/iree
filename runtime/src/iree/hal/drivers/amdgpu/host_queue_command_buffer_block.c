// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "iree/hal/drivers/amdgpu/aql_block_processor.h"
#include "iree/hal/drivers/amdgpu/aql_block_processor_profile.h"
#include "iree/hal/drivers/amdgpu/aql_command_buffer.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/device/blit.h"
#include "iree/hal/drivers/amdgpu/device/dispatch.h"
#include "iree/hal/drivers/amdgpu/device/profiling.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_internal.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_profile.h"
#include "iree/hal/drivers/amdgpu/host_queue_policy.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile.h"
#include "iree/hal/drivers/amdgpu/profile_counters.h"
#include "iree/hal/drivers/amdgpu/profile_traces.h"
#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"
#include "iree/hal/utils/resource_set.h"

static iree_status_t iree_hal_amdgpu_host_queue_resolve_buffer_ref_ptr(
    iree_hal_buffer_ref_t buffer_ref, iree_hal_buffer_usage_t required_usage,
    iree_hal_memory_access_t required_access, uint8_t** out_device_ptr) {
  *out_device_ptr = NULL;
  if (IREE_UNLIKELY(!buffer_ref.buffer)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AQL command-buffer dynamic binding resolved to a NULL buffer");
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(buffer_ref.buffer), required_usage));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(buffer_ref.buffer), required_access));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_range(
      buffer_ref.buffer, buffer_ref.offset, buffer_ref.length));
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer_ref.buffer);
  uint8_t* device_ptr =
      (uint8_t*)iree_hal_amdgpu_buffer_device_pointer(allocated_buffer);
  if (IREE_UNLIKELY(!device_ptr)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AQL command-buffer buffer must be backed by an AMDGPU allocation");
  }
  iree_device_size_t device_offset = 0;
  if (IREE_UNLIKELY(!iree_device_size_checked_add(
          iree_hal_buffer_byte_offset(buffer_ref.buffer), buffer_ref.offset,
          &device_offset))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AQL command-buffer buffer device pointer offset overflows device "
        "size");
  }
  if (IREE_UNLIKELY(device_offset > UINTPTR_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AQL command-buffer buffer device pointer offset exceeds host pointer "
        "size");
  }
  *out_device_ptr = device_ptr + (uintptr_t)device_offset;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_resolve_dispatch_binding_ptr(
    const iree_hal_buffer_binding_t* binding, uint64_t* out_binding_ptr) {
  *out_binding_ptr = 0;
  if (IREE_UNLIKELY(!binding->buffer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dispatch binding table entry is NULL");
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(binding->buffer),
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(binding->buffer),
      IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(binding->buffer),
      IREE_HAL_MEMORY_ACCESS_ANY));
  const iree_device_size_t binding_length =
      binding->length == IREE_HAL_WHOLE_BUFFER ? 0 : binding->length;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_range(
      binding->buffer, binding->offset, binding_length));

  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(binding->buffer);
  void* device_ptr = iree_hal_amdgpu_buffer_device_pointer(allocated_buffer);
  if (IREE_UNLIKELY(!device_ptr)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dispatch binding table entry must be backed by an AMDGPU allocation");
  }
  iree_device_size_t device_offset = 0;
  if (IREE_UNLIKELY(!iree_device_size_checked_add(
          iree_hal_buffer_byte_offset(binding->buffer), binding->offset,
          &device_offset))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "dispatch binding table device pointer offset overflows device size");
  }
  if (IREE_UNLIKELY(device_offset > UINTPTR_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "dispatch binding table device pointer offset exceeds host pointer "
        "size");
  }
  *out_binding_ptr = (uint64_t)((uintptr_t)device_ptr + device_offset);
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_host_queue_prepare_command_buffer_binding_ptrs(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    iree_arena_allocator_t* overflow_arena, const uint64_t** out_binding_ptrs) {
  *out_binding_ptrs = NULL;
  if (command_buffer->binding_count == 0 || block->binding_source_count == 0) {
    return iree_ok_status();
  }

  uint64_t* binding_ptrs = NULL;
  if (command_buffer->binding_count <=
      IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_BINDING_SCRATCH_CAPACITY) {
    binding_ptrs = queue->command_buffer_binding_ptr_scratch;
  } else {
    iree_host_size_t binding_ptr_bytes = 0;
    IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
        0, &binding_ptr_bytes,
        IREE_STRUCT_FIELD(command_buffer->binding_count, uint64_t, NULL)));
    IREE_TRACE_ZONE_BEGIN(z0);
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, command_buffer->binding_count);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(overflow_arena, binding_ptr_bytes,
                                (void**)&binding_ptrs));
    IREE_TRACE_ZONE_END(z0);
  }

  iree_status_t status = iree_ok_status();
  const iree_hal_amdgpu_command_buffer_binding_source_t* binding_sources =
      iree_hal_amdgpu_command_buffer_block_binding_sources_const(block);
  for (uint16_t i = 0;
       i < block->binding_source_count && iree_status_is_ok(status); ++i) {
    const iree_hal_amdgpu_command_buffer_binding_source_t* binding_source =
        &binding_sources[i];
    if (iree_any_bit_set(
            binding_source->flags,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_INDIRECT_PARAMETERS)) {
      continue;
    }
    if (!iree_all_bits_set(
            binding_source->flags,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_DYNAMIC)) {
      continue;
    }
    if (IREE_UNLIKELY(binding_source->slot >= command_buffer->binding_count)) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "command-buffer binding source slot %" PRIu32
                           " exceeds binding count %u",
                           binding_source->slot, command_buffer->binding_count);
      break;
    }
    status = iree_hal_amdgpu_host_queue_resolve_dispatch_binding_ptr(
        &binding_table.bindings[binding_source->slot],
        &binding_ptrs[binding_source->slot]);
    if (!iree_status_is_ok(status)) {
      status = iree_status_annotate_f(status, "binding_table[%" PRIu32 "]",
                                      binding_source->slot);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_binding_ptrs = binding_ptrs;
  }
  return status;
}

static iree_status_t
iree_hal_amdgpu_host_queue_prepare_command_buffer_packet_metadata(
    iree_hal_amdgpu_host_queue_t* queue, uint32_t packet_count,
    iree_arena_allocator_t* scratch_arena, uint16_t** out_packet_headers,
    uint16_t** out_packet_setups) {
  *out_packet_headers = NULL;
  *out_packet_setups = NULL;

  uint16_t* packet_headers = NULL;
  uint16_t* packet_setups = NULL;
  if (packet_count <=
      IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_SCRATCH_CAPACITY) {
    packet_headers = queue->command_buffer_packet_header_scratch;
    packet_setups = queue->command_buffer_packet_setup_scratch;
  } else {
    iree_host_size_t packet_metadata_bytes = 0;
    IREE_RETURN_IF_ERROR(
        IREE_STRUCT_LAYOUT(0, &packet_metadata_bytes,
                           IREE_STRUCT_FIELD(packet_count, uint16_t, NULL)));
    IREE_TRACE_ZONE_BEGIN(z0);
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, packet_count);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(scratch_arena, packet_metadata_bytes,
                                (void**)&packet_headers));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(scratch_arena, packet_metadata_bytes,
                                (void**)&packet_setups));
    IREE_TRACE_ZONE_END(z0);
  }

  memset(packet_headers, 0, packet_count * sizeof(*packet_headers));
  memset(packet_setups, 0, packet_count * sizeof(*packet_setups));
  *out_packet_headers = packet_headers;
  *out_packet_setups = packet_setups;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_resolve_command_buffer_ref(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_amdgpu_command_buffer_binding_kind_t kind, uint32_t ordinal,
    uint64_t offset, uint64_t length, iree_hal_buffer_usage_t required_usage,
    iree_hal_memory_access_t required_access,
    iree_hal_buffer_ref_t* out_buffer_ref, uint8_t** out_device_ptr) {
  memset(out_buffer_ref, 0, sizeof(*out_buffer_ref));
  *out_device_ptr = NULL;
  if (kind == IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_KIND_STATIC) {
    iree_hal_buffer_t* buffer =
        iree_hal_amdgpu_aql_command_buffer_static_buffer(command_buffer,
                                                         ordinal);
    if (IREE_UNLIKELY(!buffer)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AQL command-buffer static buffer ordinal %" PRIu32 " is invalid",
          ordinal);
    }
    *out_buffer_ref = iree_hal_make_buffer_ref(buffer, offset, length);
  } else if (kind == IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_KIND_DYNAMIC) {
    iree_hal_buffer_ref_t dynamic_ref =
        iree_hal_make_indirect_buffer_ref(ordinal, offset, length);
    IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
        binding_table, dynamic_ref, out_buffer_ref));
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AQL command-buffer binding kind %u is invalid",
                            kind);
  }
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_resolve_buffer_ref_ptr(
      *out_buffer_ref, required_usage, required_access, out_device_ptr));
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_host_queue_resolve_static_binding_source_ptr(
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_command_buffer_binding_source_t* binding_source,
    uint64_t* out_binding_ptr) {
  *out_binding_ptr = 0;
  iree_hal_buffer_t* buffer = iree_hal_amdgpu_aql_command_buffer_static_buffer(
      command_buffer, binding_source->slot);
  if (IREE_UNLIKELY(!buffer)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AQL command-buffer static dispatch binding ordinal %" PRIu32
        " is invalid",
        binding_source->slot);
  }
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer);
  void* device_ptr = iree_hal_amdgpu_buffer_device_pointer(allocated_buffer);
  if (IREE_UNLIKELY(!device_ptr)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AQL command-buffer static dispatch binding has no staged AMDGPU "
        "backing after queue waits completed");
  }
  iree_device_size_t device_offset = 0;
  if (IREE_UNLIKELY(!iree_device_size_checked_add(
          iree_hal_buffer_byte_offset(buffer),
          binding_source->offset_or_pointer, &device_offset))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AQL command-buffer static dispatch binding pointer offset overflows "
        "device size");
  }
  if (IREE_UNLIKELY(device_offset > UINTPTR_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AQL command-buffer static dispatch binding pointer offset exceeds "
        "host pointer size");
  }
  *out_binding_ptr =
      (uint64_t)((uintptr_t)device_ptr + (uintptr_t)device_offset);
  return iree_ok_status();
}

static bool iree_hal_amdgpu_host_queue_command_buffer_packet_has_barrier(
    const iree_hal_amdgpu_wait_resolution_t* resolution, uint32_t packet_index,
    iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t packet_flags) {
  return iree_any_bit_set(
             packet_flags,
             IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_EXECUTION_BARRIER) ||
         iree_any_bit_set(
             packet_flags,
             IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_FINAL) ||
         (packet_index == 0 && resolution->barrier_count > 0) ||
         (packet_index == 0 &&
          resolution->inline_acquire_scope != IREE_HSA_FENCE_SCOPE_NONE);
}

static iree_hsa_fence_scope_t
iree_hal_amdgpu_host_queue_command_buffer_block_payload_acquire_scope(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_command_buffer_block_header_t* block) {
  if (block->kernarg_length == 0) return IREE_HSA_FENCE_SCOPE_NONE;
  return iree_hal_amdgpu_host_queue_kernarg_acquire_scope(
      queue, IREE_HSA_FENCE_SCOPE_NONE);
}

static uint32_t
iree_hal_amdgpu_host_queue_command_buffer_block_payload_acquire_packet_count(
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    uint32_t packet_index_base, iree_hsa_fence_scope_t payload_acquire_scope) {
  if (payload_acquire_scope == IREE_HSA_FENCE_SCOPE_NONE) return 0;
  if (block->aql_packet_count == 0) return 0;
  if (iree_hal_amdgpu_host_queue_command_buffer_packet_has_barrier(
          resolution, packet_index_base,
          block->aql_packet_count == 1
              ? IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_FINAL
              : IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_NONE)) {
    return 1;
  }
  return block->initial_barrier_packet_count;
}

static iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t
iree_hal_amdgpu_host_queue_replay_command_packet_flags(
    const iree_hal_amdgpu_command_buffer_command_header_t* command) {
  iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t packet_flags =
      IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_NONE;
  if (iree_any_bit_set(
          command->flags,
          IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_HAS_BARRIER)) {
    packet_flags |=
        IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_EXECUTION_BARRIER;
  }
  return iree_hal_amdgpu_host_queue_command_buffer_packet_flags_set_fence_scopes(
      packet_flags,
      (iree_hsa_fence_scope_t)
          iree_hal_amdgpu_command_buffer_command_flags_acquire_scope(
              command->flags),
      (iree_hsa_fence_scope_t)
          iree_hal_amdgpu_command_buffer_command_flags_release_scope(
              command->flags));
}

static iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t
iree_hal_amdgpu_host_queue_command_buffer_packet_flags_merge(
    iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t lhs,
    iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t rhs) {
  const iree_hsa_fence_scope_t acquire_scope =
      iree_hal_amdgpu_host_queue_max_fence_scope(
          iree_hal_amdgpu_host_queue_command_buffer_packet_flags_acquire_scope(
              lhs),
          iree_hal_amdgpu_host_queue_command_buffer_packet_flags_acquire_scope(
              rhs));
  const iree_hsa_fence_scope_t release_scope =
      iree_hal_amdgpu_host_queue_max_fence_scope(
          iree_hal_amdgpu_host_queue_command_buffer_packet_flags_release_scope(
              lhs),
          iree_hal_amdgpu_host_queue_command_buffer_packet_flags_release_scope(
              rhs));
  return iree_hal_amdgpu_host_queue_command_buffer_packet_flags_set_fence_scopes(
      lhs | rhs, acquire_scope, release_scope);
}

static iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t
iree_hal_amdgpu_host_queue_command_buffer_agent_barrier_packet_flags(void) {
  return iree_hal_amdgpu_host_queue_command_buffer_packet_flags_set_fence_scopes(
      IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_EXECUTION_BARRIER,
      IREE_HSA_FENCE_SCOPE_AGENT, IREE_HSA_FENCE_SCOPE_AGENT);
}

static bool iree_hal_amdgpu_host_queue_replay_command_uses_queue_kernargs(
    const iree_hal_amdgpu_command_buffer_command_header_t* command) {
  return iree_any_bit_set(
      command->flags,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS);
}

static iree_hsa_fence_scope_t
iree_hal_amdgpu_host_queue_command_buffer_payload_acquire_scope(
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    iree_hsa_fence_scope_t payload_acquire_scope,
    uint32_t payload_acquire_packet_count, uint32_t logical_packet_index,
    uint32_t recorded_packet_index,
    const iree_hal_amdgpu_command_buffer_command_header_t* command,
    iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t packet_flags) {
  if (payload_acquire_scope == IREE_HSA_FENCE_SCOPE_NONE) {
    return IREE_HSA_FENCE_SCOPE_NONE;
  }
  if (recorded_packet_index >= payload_acquire_packet_count) {
    return IREE_HSA_FENCE_SCOPE_NONE;
  }
  if (iree_hal_amdgpu_host_queue_replay_command_uses_queue_kernargs(command)) {
    return payload_acquire_scope;
  }
  if (iree_hal_amdgpu_host_queue_command_buffer_packet_has_barrier(
          resolution, logical_packet_index, packet_flags)) {
    return payload_acquire_scope;
  }
  return IREE_HSA_FENCE_SCOPE_NONE;
}

static bool iree_hal_amdgpu_host_queue_dispatch_uses_indirect_parameters(
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command) {
  return iree_any_bit_set(
      dispatch_command->dispatch_flags,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_DISPATCH_FLAG_INDIRECT_PARAMETERS);
}

static uint32_t iree_hal_amdgpu_host_queue_aql_packet_header_field(
    uint16_t header, uint32_t field_shift, uint32_t field_width) {
  return (header >> field_shift) & ((1u << field_width) - 1u);
}

static iree_hsa_packet_type_t iree_hal_amdgpu_host_queue_aql_packet_header_type(
    uint16_t header) {
  return (iree_hsa_packet_type_t)
      iree_hal_amdgpu_host_queue_aql_packet_header_field(
          header, IREE_HSA_PACKET_HEADER_TYPE,
          IREE_HSA_PACKET_HEADER_WIDTH_TYPE);
}

static bool iree_hal_amdgpu_host_queue_aql_packet_header_has_barrier(
    uint16_t header) {
  return iree_hal_amdgpu_host_queue_aql_packet_header_field(
             header, IREE_HSA_PACKET_HEADER_BARRIER,
             IREE_HSA_PACKET_HEADER_WIDTH_BARRIER) != 0;
}

static iree_hsa_fence_scope_t
iree_hal_amdgpu_host_queue_aql_packet_header_acquire_scope(uint16_t header) {
  return (iree_hsa_fence_scope_t)
      iree_hal_amdgpu_host_queue_aql_packet_header_field(
          header, IREE_HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE,
          IREE_HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE);
}

static iree_hsa_fence_scope_t
iree_hal_amdgpu_host_queue_aql_packet_header_release_scope(uint16_t header) {
  return (iree_hsa_fence_scope_t)
      iree_hal_amdgpu_host_queue_aql_packet_header_field(
          header, IREE_HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE,
          IREE_HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE);
}

static bool iree_hal_amdgpu_host_queue_dispatch_uses_prepublished_kernargs(
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command) {
  return dispatch_command->kernarg_strategy ==
         IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_PREPUBLISHED;
}

static bool iree_hal_amdgpu_host_queue_dispatch_uses_queue_kernargs(
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command) {
  return !iree_hal_amdgpu_host_queue_dispatch_uses_prepublished_kernargs(
      dispatch_command);
}

static uint32_t iree_hal_amdgpu_host_queue_dispatch_target_kernarg_block_count(
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command) {
  if (iree_hal_amdgpu_host_queue_dispatch_uses_prepublished_kernargs(
          dispatch_command)) {
    return 0;
  }
  const uint32_t kernarg_length =
      (uint32_t)dispatch_command->kernarg_length_qwords * 8u;
  return iree_max(1u,
                  (uint32_t)iree_host_size_ceil_div(
                      kernarg_length, sizeof(iree_hal_amdgpu_kernarg_block_t)));
}

static uint32_t iree_hal_amdgpu_host_queue_dispatch_kernarg_block_count(
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command) {
  return iree_hal_amdgpu_host_queue_dispatch_target_kernarg_block_count(
             dispatch_command) +
         (iree_hal_amdgpu_host_queue_dispatch_uses_indirect_parameters(
              dispatch_command)
              ? 1u
              : 0u);
}

static void
iree_hal_amdgpu_host_queue_write_command_buffer_dispatch_packet_body(
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command,
    iree_hal_amdgpu_aql_packet_t* packet, uint8_t* kernarg_data,
    iree_hsa_signal_t completion_signal, uint16_t* out_setup) {
  packet->dispatch.setup = dispatch_command->setup;
  packet->dispatch.workgroup_size[0] = dispatch_command->workgroup_size[0];
  packet->dispatch.workgroup_size[1] = dispatch_command->workgroup_size[1];
  packet->dispatch.workgroup_size[2] = dispatch_command->workgroup_size[2];
  packet->dispatch.reserved0 = 0;
  packet->dispatch.grid_size[0] = dispatch_command->grid_size[0];
  packet->dispatch.grid_size[1] = dispatch_command->grid_size[1];
  packet->dispatch.grid_size[2] = dispatch_command->grid_size[2];
  packet->dispatch.private_segment_size =
      dispatch_command->private_segment_size;
  packet->dispatch.group_segment_size = dispatch_command->group_segment_size;
  packet->dispatch.kernel_object = dispatch_command->kernel_object;
  packet->dispatch.kernarg_address = kernarg_data;
  packet->dispatch.reserved2 = 0;
  packet->dispatch.completion_signal = completion_signal;
  *out_setup = packet->dispatch.setup;
}

static iree_status_t iree_hal_amdgpu_host_queue_replay_dispatch_kernargs(
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    iree_hal_command_buffer_t* command_buffer, const uint64_t* binding_ptrs,
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command,
    uint8_t* kernarg_data) {
  const uint8_t* command_base = (const uint8_t*)dispatch_command;
  const uint8_t* tail_payload =
      command_base + dispatch_command->payload_reference;
  const iree_host_size_t tail_length =
      (iree_host_size_t)dispatch_command->tail_length_qwords * 8u;

  switch (dispatch_command->kernarg_strategy) {
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_HAL: {
      uint64_t* binding_dst = (uint64_t*)kernarg_data;
      const iree_hal_amdgpu_command_buffer_binding_source_t* binding_sources =
          (const iree_hal_amdgpu_command_buffer_binding_source_t*)((const uint8_t*)
                                                                       block +
                                                                   dispatch_command
                                                                       ->binding_source_offset);
      for (uint16_t i = 0; i < dispatch_command->binding_count; ++i) {
        const iree_hal_amdgpu_command_buffer_binding_source_t* binding_source =
            &binding_sources[i];
        const uint32_t flags = binding_source->flags;
        if (IREE_LIKELY(
                flags ==
                IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_NONE)) {
          binding_dst[i] = binding_source->offset_or_pointer;
        } else if (flags ==
                   IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_DYNAMIC) {
          if (IREE_UNLIKELY(!binding_ptrs)) {
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "AQL command-buffer dispatch has dynamic bindings but no "
                "binding table was provided");
          }
          binding_dst[i] = binding_ptrs[binding_source->slot] +
                           binding_source->offset_or_pointer;
        } else if (
            flags ==
            IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_STATIC_BUFFER) {
          IREE_RETURN_IF_ERROR(
              iree_hal_amdgpu_host_queue_resolve_static_binding_source_ptr(
                  command_buffer, binding_source, &binding_dst[i]));
        } else {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "malformed AQL command-buffer dispatch binding source flags %u",
              binding_source->flags);
        }
      }
      if (tail_length > 0) {
        memcpy(
            kernarg_data + (iree_host_size_t)dispatch_command->binding_count *
                               sizeof(uint64_t),
            tail_payload, tail_length);
      }
      return iree_ok_status();
    }
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_CUSTOM_DIRECT:
      if (tail_length > 0) {
        memcpy(kernarg_data, tail_payload, tail_length);
      }
      return iree_ok_status();
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_INDIRECT:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "indirect dispatch arguments are not supported by AMDGPU command "
          "buffers yet");
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_PREPUBLISHED:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "prepublished command-buffer dispatch should not rewrite kernargs");
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "malformed AQL command-buffer kernarg strategy "
                              "%u",
                              dispatch_command->kernarg_strategy);
  }
}

static iree_status_t
iree_hal_amdgpu_host_queue_replay_dispatch_indirect_params_ptr(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_amdgpu_command_buffer_binding_source_t* binding_source,
    const uint32_t** out_workgroup_count_ptr) {
  *out_workgroup_count_ptr = NULL;
  switch (binding_source->flags) {
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_INDIRECT_PARAMETERS:
      *out_workgroup_count_ptr =
          (const uint32_t*)(uintptr_t)binding_source->offset_or_pointer;
      return iree_ok_status();
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_DYNAMIC |
        IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_INDIRECT_PARAMETERS: {
      iree_hal_buffer_ref_t resolved_ref = {0};
      IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
          binding_table,
          iree_hal_make_indirect_buffer_ref(binding_source->slot,
                                            binding_source->offset_or_pointer,
                                            sizeof(uint32_t[3])),
          &resolved_ref));
      uint8_t* device_ptr = NULL;
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_resolve_buffer_ref_ptr(
          resolved_ref, IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS,
          IREE_HAL_MEMORY_ACCESS_READ, &device_ptr));
      *out_workgroup_count_ptr = (const uint32_t*)device_ptr;
      return iree_ok_status();
    }
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_STATIC_BUFFER |
        IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_INDIRECT_PARAMETERS: {
      uint64_t workgroup_count_ptr = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_host_queue_resolve_static_binding_source_ptr(
              command_buffer, binding_source, &workgroup_count_ptr));
      *out_workgroup_count_ptr =
          (const uint32_t*)(uintptr_t)workgroup_count_ptr;
      return iree_ok_status();
    }
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "malformed AQL command-buffer indirect parameter source flags %u",
          binding_source->flags);
  }
}

static iree_amdgpu_kernel_implicit_args_t*
iree_hal_amdgpu_host_queue_dispatch_implicit_args_ptr(
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command,
    uint8_t* kernarg_data) {
  if (dispatch_command->implicit_args_offset_qwords == UINT16_MAX) {
    return NULL;
  }
  return (
      iree_amdgpu_kernel_implicit_args_t*)(kernarg_data +
                                           (iree_host_size_t)dispatch_command
                                                   ->implicit_args_offset_qwords *
                                               8u);
}

static iree_status_t iree_hal_amdgpu_host_queue_replay_dispatch_packet_body(
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    iree_hal_command_buffer_t* command_buffer, const uint64_t* binding_ptrs,
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command,
    iree_hal_amdgpu_aql_packet_t* packet, uint8_t* kernarg_data,
    iree_hsa_signal_t completion_signal, uint16_t* out_setup) {
  if (iree_hal_amdgpu_host_queue_dispatch_uses_prepublished_kernargs(
          dispatch_command)) {
    const uint32_t kernarg_length =
        (uint32_t)dispatch_command->kernarg_length_qwords * 8u;
    kernarg_data = iree_hal_amdgpu_aql_command_buffer_prepublished_kernarg(
        command_buffer, dispatch_command->payload_reference, kernarg_length);
    if (IREE_UNLIKELY(!kernarg_data)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AQL command-buffer prepublished kernarg range is invalid");
    }
  } else {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_replay_dispatch_kernargs(
        block, command_buffer, binding_ptrs, dispatch_command, kernarg_data));
  }
  iree_hal_amdgpu_host_queue_write_command_buffer_dispatch_packet_body(
      dispatch_command, packet, kernarg_data, completion_signal, out_setup);
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_host_queue_replay_indirect_dispatch_packet_bodies(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table, const uint64_t* binding_ptrs,
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command,
    iree_hal_amdgpu_aql_packet_t* patch_packet,
    iree_hal_amdgpu_aql_packet_t* dispatch_packet, uint8_t* patch_kernarg_data,
    uint8_t* dispatch_kernarg_data, iree_hsa_signal_t completion_signal,
    uint16_t dispatch_header, uint16_t* out_patch_setup,
    uint16_t* out_dispatch_setup) {
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_replay_dispatch_packet_body(
      block, command_buffer, binding_ptrs, dispatch_command, dispatch_packet,
      dispatch_kernarg_data, completion_signal, out_dispatch_setup));

  const iree_hal_amdgpu_command_buffer_binding_source_t* binding_sources =
      (const iree_hal_amdgpu_command_buffer_binding_source_t*)((const uint8_t*)
                                                                   block +
                                                               dispatch_command
                                                                   ->binding_source_offset);
  const iree_hal_amdgpu_command_buffer_binding_source_t*
      indirect_params_source =
          &binding_sources[dispatch_command->binding_count];
  const uint32_t* workgroup_count_ptr = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_replay_dispatch_indirect_params_ptr(
          command_buffer, binding_table, indirect_params_source,
          &workgroup_count_ptr));

  iree_amdgpu_kernel_implicit_args_t* implicit_args =
      iree_hal_amdgpu_host_queue_dispatch_implicit_args_ptr(
          dispatch_command, dispatch_kernarg_data);
  iree_hal_amdgpu_device_dispatch_emplace_indirect_params_patch(
      &queue->transfer_context->kernels
           ->iree_hal_amdgpu_device_dispatch_patch_indirect_params,
      workgroup_count_ptr, &dispatch_packet->dispatch, dispatch_header,
      *out_dispatch_setup, implicit_args, &patch_packet->dispatch,
      patch_kernarg_data);
  *out_patch_setup = patch_packet->dispatch.setup;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_replay_fill_packet_body(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_amdgpu_command_buffer_fill_command_t* fill_command,
    iree_hal_amdgpu_aql_packet_t* packet,
    iree_hal_amdgpu_kernarg_block_t* kernarg_block,
    iree_hsa_signal_t completion_signal, uint16_t* out_setup) {
  iree_hal_buffer_ref_t target_ref = {0};
  uint8_t* target_ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_resolve_command_buffer_ref(
      command_buffer, binding_table, fill_command->target_kind,
      fill_command->target_ordinal, fill_command->target_offset,
      fill_command->length, IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET,
      IREE_HAL_MEMORY_ACCESS_WRITE, &target_ref, &target_ptr));
  (void)target_ref;
  if (IREE_UNLIKELY(!iree_hal_amdgpu_device_buffer_fill_emplace(
          queue->transfer_context, &packet->dispatch, target_ptr,
          fill_command->length, fill_command->pattern,
          fill_command->pattern_length, kernarg_block->data))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported command-buffer fill dispatch shape");
  }
  packet->dispatch.completion_signal = completion_signal;
  *out_setup = packet->dispatch.setup;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_replay_copy_packet_body(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_amdgpu_command_buffer_copy_command_t* copy_command,
    iree_hal_amdgpu_aql_packet_t* packet,
    iree_hal_amdgpu_kernarg_block_t* kernarg_block,
    iree_hsa_signal_t completion_signal, uint16_t* out_setup) {
  iree_hal_buffer_ref_t source_ref = {0};
  uint8_t* source_ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_resolve_command_buffer_ref(
      command_buffer, binding_table, copy_command->source_kind,
      copy_command->source_ordinal, copy_command->source_offset,
      copy_command->length, IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE,
      IREE_HAL_MEMORY_ACCESS_READ, &source_ref, &source_ptr));
  iree_hal_buffer_ref_t target_ref = {0};
  uint8_t* target_ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_resolve_command_buffer_ref(
      command_buffer, binding_table, copy_command->target_kind,
      copy_command->target_ordinal, copy_command->target_offset,
      copy_command->length, IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET,
      IREE_HAL_MEMORY_ACCESS_WRITE, &target_ref, &target_ptr));

  if (IREE_UNLIKELY(
          iree_hal_buffer_test_overlap(source_ref.buffer, source_ref.offset,
                                       source_ref.length, target_ref.buffer,
                                       target_ref.offset, target_ref.length) !=
          IREE_HAL_BUFFER_OVERLAP_DISJOINT)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source and target ranges must not overlap within the same buffer");
  }
  if (IREE_UNLIKELY(!iree_hal_amdgpu_device_buffer_copy_emplace(
          queue->transfer_context, &packet->dispatch, source_ptr, target_ptr,
          copy_command->length, kernarg_block->data))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported command-buffer copy dispatch shape");
  }
  packet->dispatch.completion_signal = completion_signal;
  *out_setup = packet->dispatch.setup;
  return iree_ok_status();
}

static iree_host_size_t iree_hal_amdgpu_host_queue_update_kernarg_length(
    uint32_t source_length) {
  const iree_host_size_t source_payload_offset =
      IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_STAGED_SOURCE_OFFSET;
  return source_payload_offset + (iree_host_size_t)source_length;
}

static uint32_t iree_hal_amdgpu_host_queue_update_kernarg_block_count(
    uint32_t source_length) {
  return (uint32_t)iree_host_size_ceil_div(
      iree_hal_amdgpu_host_queue_update_kernarg_length(source_length),
      sizeof(iree_hal_amdgpu_kernarg_block_t));
}

static iree_status_t iree_hal_amdgpu_host_queue_resolve_update_packet_operands(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_amdgpu_command_buffer_update_command_t* update_command,
    const uint8_t** out_source_bytes, uint8_t** out_target_ptr) {
  *out_source_bytes = NULL;
  *out_target_ptr = NULL;
  iree_hal_buffer_ref_t target_ref = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_resolve_command_buffer_ref(
      command_buffer, binding_table, update_command->target_kind,
      update_command->target_ordinal, update_command->target_offset,
      update_command->length, IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET,
      IREE_HAL_MEMORY_ACCESS_WRITE, &target_ref, out_target_ptr));
  (void)target_ref;
  *out_source_bytes = iree_hal_amdgpu_aql_command_buffer_rodata(
      command_buffer, update_command->rodata_ordinal, update_command->length);
  if (IREE_UNLIKELY(!*out_source_bytes)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AQL command-buffer update rodata range is invalid");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_replay_update_packet_body(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_amdgpu_command_buffer_update_command_t* update_command,
    iree_hal_amdgpu_aql_packet_t* packet, uint8_t* kernarg_data,
    iree_host_size_t kernarg_length, iree_hsa_signal_t completion_signal,
    uint16_t* out_setup) {
  const uint8_t* source_bytes = NULL;
  uint8_t* target_ptr = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_resolve_update_packet_operands(
          command_buffer, binding_table, update_command, &source_bytes,
          &target_ptr));

  const iree_host_size_t source_payload_offset =
      IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_STAGED_SOURCE_OFFSET;
  const iree_host_size_t required_kernarg_length =
      source_payload_offset + (iree_host_size_t)update_command->length;
  if (IREE_UNLIKELY(required_kernarg_length > kernarg_length)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AQL command-buffer update kernarg range is too small");
  }

  iree_hal_amdgpu_device_buffer_copy_kernargs_t kernargs;
  memset(&kernargs, 0, sizeof(kernargs));
  if (IREE_UNLIKELY(!iree_hal_amdgpu_device_buffer_copy_emplace(
          queue->transfer_context, &packet->dispatch,
          (const void*)(uintptr_t)
              IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_STAGED_SOURCE_ALIGNMENT,
          target_ptr, update_command->length, &kernargs))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported command-buffer update dispatch shape");
  }

  uint8_t* staged_source_bytes = kernarg_data + source_payload_offset;
  memcpy(kernarg_data, &kernargs, sizeof(kernargs));
  ((iree_hal_amdgpu_device_buffer_copy_kernargs_t*)kernarg_data)->source_ptr =
      staged_source_bytes;
  memcpy(staged_source_bytes, source_bytes, update_command->length);
  packet->dispatch.kernarg_address = kernarg_data;
  packet->dispatch.completion_signal = completion_signal;
  *out_setup = packet->dispatch.setup;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_host_queue_validate_metadata_commands(
    const iree_hal_amdgpu_aql_program_t* program) {
  const iree_hal_amdgpu_command_buffer_block_header_t* block =
      program->first_block;
  bool reached_return = false;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) && !reached_return && block) {
    const iree_hal_amdgpu_command_buffer_command_header_t* command =
        iree_hal_amdgpu_command_buffer_block_commands_const(block);
    bool advanced_block = false;
    for (uint16_t i = 0;
         i < block->command_count && iree_status_is_ok(status) &&
         !reached_return && !advanced_block;
         ++i) {
      switch (command->opcode) {
        case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BARRIER:
          break;
        case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH: {
          const iree_hal_amdgpu_command_buffer_branch_command_t*
              branch_command =
                  (const iree_hal_amdgpu_command_buffer_branch_command_t*)
                      command;
          iree_hal_amdgpu_command_buffer_block_header_t* next_block =
              iree_hal_amdgpu_aql_program_block_next(program->block_pool,
                                                     block);
          if (IREE_UNLIKELY(!next_block ||
                            branch_command->target_block_ordinal !=
                                next_block->block_ordinal)) {
            status = iree_make_status(
                IREE_STATUS_UNIMPLEMENTED,
                "non-linear AQL command-buffer branch replay not yet wired");
          } else {
            block = next_block;
            advanced_block = true;
          }
          break;
        }
        case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN:
          reached_return = true;
          break;
        case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH:
        case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_FILL:
        case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COPY:
        case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_UPDATE:
        case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_PROFILE_MARKER:
        case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COND_BRANCH:
          status = iree_make_status(
              IREE_STATUS_UNIMPLEMENTED,
              "AQL command-buffer opcode %u replay not yet wired",
              command->opcode);
          break;
        default:
          status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "malformed AQL command-buffer opcode %u",
                                    command->opcode);
          break;
      }
      if (iree_status_is_ok(status) && !reached_return && !advanced_block) {
        command = iree_hal_amdgpu_command_buffer_command_next_const(command);
      }
    }
    if (iree_status_is_ok(status) && !reached_return && !advanced_block) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AQL command-buffer block %" PRIu32
                                " has no terminator",
                                block->block_ordinal);
    }
  }
  if (iree_status_is_ok(status) && !reached_return) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AQL command-buffer program has no return");
  }
  return status;
}

#if !defined(NDEBUG)
static void iree_hal_amdgpu_host_queue_accumulate_packet_summary(
    iree_hal_amdgpu_host_queue_command_buffer_packet_summary_t* summary,
    uint16_t header) {
  if (summary->packet_count == 0) summary->first_packet_header = header;
  summary->last_packet_header = header;
  ++summary->packet_count;
  if (iree_hal_amdgpu_host_queue_aql_packet_header_has_barrier(header)) {
    ++summary->barrier_packet_count;
  }
  if (iree_hal_amdgpu_host_queue_aql_packet_header_acquire_scope(header) ==
      IREE_HSA_FENCE_SCOPE_SYSTEM) {
    ++summary->system_acquire_packet_count;
  }
  if (iree_hal_amdgpu_host_queue_aql_packet_header_release_scope(header) ==
      IREE_HSA_FENCE_SCOPE_SYSTEM) {
    ++summary->system_release_packet_count;
  }
}

iree_status_t iree_hal_amdgpu_host_queue_summarize_command_buffer_block_packets(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    iree_hal_amdgpu_host_queue_command_buffer_packet_summary_t* out_summary) {
  memset(out_summary, 0, sizeof(*out_summary));

  const iree_hal_amdgpu_command_buffer_command_header_t* command =
      iree_hal_amdgpu_command_buffer_block_commands_const(block);
  const iree_hsa_fence_scope_t payload_acquire_scope =
      iree_hal_amdgpu_host_queue_command_buffer_block_payload_acquire_scope(
          queue, block);
  const uint32_t payload_acquire_packet_count =
      iree_hal_amdgpu_host_queue_command_buffer_block_payload_acquire_packet_count(
          resolution, block, /*packet_index_base=*/0, payload_acquire_scope);
  bool reached_terminator = false;
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < block->command_count && iree_status_is_ok(status) &&
                       !reached_terminator;
       ++i) {
    switch (command->opcode) {
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BARRIER:
        break;
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH: {
        const iree_hal_amdgpu_command_buffer_dispatch_command_t*
            dispatch_command =
                (const iree_hal_amdgpu_command_buffer_dispatch_command_t*)
                    command;
        iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t packet_flags =
            iree_hal_amdgpu_host_queue_replay_command_packet_flags(command);
        const iree_hsa_fence_scope_t first_packet_acquire_scope =
            iree_hal_amdgpu_host_queue_command_buffer_payload_acquire_scope(
                resolution, payload_acquire_scope, payload_acquire_packet_count,
                out_summary->packet_count, out_summary->packet_count, command,
                packet_flags);
        if (iree_hal_amdgpu_host_queue_dispatch_uses_indirect_parameters(
                dispatch_command)) {
          // The patch dispatch publishes the following dispatch packet header.
          // It must retire before the CP observes that following slot.
          const uint16_t patch_header = iree_hal_amdgpu_aql_make_header(
              IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
              iree_hal_amdgpu_host_queue_command_buffer_packet_control(
                  queue, resolution, signal_semaphore_list,
                  out_summary->packet_count, first_packet_acquire_scope,
                  packet_flags));
          iree_hal_amdgpu_host_queue_accumulate_packet_summary(out_summary,
                                                               patch_header);

          packet_flags =
              IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_NONE;
          if (out_summary->packet_count + 1 == block->aql_packet_count) {
            packet_flags |=
                IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_FINAL;
          }
          const uint16_t dispatch_header = iree_hal_amdgpu_aql_make_header(
              IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
              iree_hal_amdgpu_host_queue_command_buffer_packet_control(
                  queue, resolution, signal_semaphore_list,
                  out_summary->packet_count,
                  iree_hal_amdgpu_host_queue_command_buffer_payload_acquire_scope(
                      resolution, payload_acquire_scope,
                      payload_acquire_packet_count, out_summary->packet_count,
                      out_summary->packet_count, command, packet_flags),
                  packet_flags));
          iree_hal_amdgpu_host_queue_accumulate_packet_summary(out_summary,
                                                               dispatch_header);
        } else {
          if (out_summary->packet_count + 1 == block->aql_packet_count) {
            packet_flags |=
                IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_FINAL;
          }
          const uint16_t header = iree_hal_amdgpu_aql_make_header(
              IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
              iree_hal_amdgpu_host_queue_command_buffer_packet_control(
                  queue, resolution, signal_semaphore_list,
                  out_summary->packet_count, first_packet_acquire_scope,
                  packet_flags));
          iree_hal_amdgpu_host_queue_accumulate_packet_summary(out_summary,
                                                               header);
        }
        break;
      }
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_FILL:
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COPY:
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_UPDATE: {
        iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t packet_flags =
            iree_hal_amdgpu_host_queue_replay_command_packet_flags(command);
        if (out_summary->packet_count + 1 == block->aql_packet_count) {
          packet_flags |=
              IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_FINAL;
        }
        const uint16_t header = iree_hal_amdgpu_aql_make_header(
            IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
            iree_hal_amdgpu_host_queue_command_buffer_packet_control(
                queue, resolution, signal_semaphore_list,
                out_summary->packet_count,
                iree_hal_amdgpu_host_queue_command_buffer_payload_acquire_scope(
                    resolution, payload_acquire_scope,
                    payload_acquire_packet_count, out_summary->packet_count,
                    out_summary->packet_count, command, packet_flags),
                packet_flags));
        iree_hal_amdgpu_host_queue_accumulate_packet_summary(out_summary,
                                                             header);
        break;
      }
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN:
        reached_terminator = true;
        break;
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH:
        reached_terminator = true;
        break;
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_PROFILE_MARKER:
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COND_BRANCH:
        status = iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "AQL command-buffer opcode %u replay not yet wired",
            command->opcode);
        break;
      default:
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "malformed AQL command-buffer opcode %u",
                                  command->opcode);
        break;
    }
    if (iree_status_is_ok(status) && !reached_terminator) {
      command = iree_hal_amdgpu_command_buffer_command_next_const(command);
    }
  }
  if (iree_status_is_ok(status) && !reached_terminator) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AQL command-buffer block %" PRIu32
                              " has no terminator",
                              block->block_ordinal);
  }
  if (iree_status_is_ok(status) &&
      out_summary->packet_count != block->aql_packet_count) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AQL command-buffer block %" PRIu32
                              " summarizes %" PRIu32
                              " packets but declares %" PRIu32,
                              block->block_ordinal, out_summary->packet_count,
                              block->aql_packet_count);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_check_update_packet_command(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_amdgpu_command_buffer_update_command_t* update_command,
    iree_hal_amdgpu_aql_packet_t* packet, uint16_t* out_setup) {
  const uint8_t* source_bytes = NULL;
  uint8_t* target_ptr = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_resolve_update_packet_operands(
          command_buffer, binding_table, update_command, &source_bytes,
          &target_ptr));
  (void)source_bytes;

  iree_hal_amdgpu_device_buffer_copy_kernargs_t kernargs;
  memset(&kernargs, 0, sizeof(kernargs));
  if (IREE_UNLIKELY(!iree_hal_amdgpu_device_buffer_copy_emplace(
          queue->transfer_context, &packet->dispatch,
          (const void*)(uintptr_t)
              IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_STAGED_SOURCE_ALIGNMENT,
          target_ptr, update_command->length, &kernargs))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported command-buffer update dispatch shape");
  }
  packet->dispatch.completion_signal = iree_hsa_signal_null();
  *out_setup = packet->dispatch.setup;
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_host_queue_prepare_check_dispatch_kernarg_data(
    iree_arena_allocator_t* scratch_arena,
    const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command,
    iree_hal_amdgpu_kernarg_block_t* single_block, uint8_t** out_kernarg_data) {
  *out_kernarg_data = NULL;
  const uint32_t kernarg_block_count =
      iree_hal_amdgpu_host_queue_dispatch_target_kernarg_block_count(
          dispatch_command);
  if (kernarg_block_count == 1) {
    memset(single_block, 0, sizeof(*single_block));
    *out_kernarg_data = single_block->data;
    return iree_ok_status();
  }

  iree_host_size_t kernarg_length = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &kernarg_length,
      IREE_STRUCT_FIELD(kernarg_block_count, iree_hal_amdgpu_kernarg_block_t,
                        NULL)));
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, kernarg_block_count);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(scratch_arena, kernarg_length,
                              (void**)out_kernarg_data));
  IREE_TRACE_ZONE_END(z0);
  memset(*out_kernarg_data, 0, kernarg_length);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_check_packet_commands(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table, const uint64_t* binding_ptrs,
    iree_arena_allocator_t* scratch_arena,
    const iree_hal_amdgpu_command_buffer_block_header_t* block) {
  iree_hal_amdgpu_aql_packet_t packet;
  iree_hal_amdgpu_kernarg_block_t kernarg_block;
  memset(&packet, 0, sizeof(packet));

  const iree_hal_amdgpu_command_buffer_command_header_t* command =
      iree_hal_amdgpu_command_buffer_block_commands_const(block);
  bool reached_terminator = false;
  uint32_t packet_count = 0;
  uint32_t kernarg_block_count = 0;
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < block->command_count && iree_status_is_ok(status) &&
                       !reached_terminator;
       ++i) {
    uint16_t setup = 0;
    switch (command->opcode) {
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BARRIER:
        break;
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH: {
        const iree_hal_amdgpu_command_buffer_dispatch_command_t*
            dispatch_command =
                (const iree_hal_amdgpu_command_buffer_dispatch_command_t*)
                    command;
        if (iree_hal_amdgpu_host_queue_dispatch_uses_indirect_parameters(
                dispatch_command)) {
          iree_hal_amdgpu_aql_packet_t dispatch_packet;
          iree_hal_amdgpu_kernarg_block_t patch_kernarg_block;
          uint16_t dispatch_setup = 0;
          uint16_t patch_setup = 0;
          uint8_t* dispatch_kernarg_data = NULL;
          status =
              iree_hal_amdgpu_host_queue_prepare_check_dispatch_kernarg_data(
                  scratch_arena, dispatch_command, &kernarg_block,
                  &dispatch_kernarg_data);
          if (iree_status_is_ok(status)) {
            const uint16_t dispatch_header = iree_hal_amdgpu_aql_make_header(
                IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
                iree_hal_amdgpu_aql_packet_control(
                    /*has_barrier=*/false, IREE_HSA_FENCE_SCOPE_AGENT,
                    IREE_HSA_FENCE_SCOPE_AGENT));
            status =
                iree_hal_amdgpu_host_queue_replay_indirect_dispatch_packet_bodies(
                    queue, block, command_buffer, binding_table, binding_ptrs,
                    dispatch_command, &packet, &dispatch_packet,
                    patch_kernarg_block.data, dispatch_kernarg_data,
                    iree_hsa_signal_null(), dispatch_header, &patch_setup,
                    &dispatch_setup);
          }
          if (iree_status_is_ok(status)) packet_count += 2;
        } else {
          uint8_t* kernarg_data = NULL;
          if (iree_hal_amdgpu_host_queue_dispatch_uses_queue_kernargs(
                  dispatch_command)) {
            status =
                iree_hal_amdgpu_host_queue_prepare_check_dispatch_kernarg_data(
                    scratch_arena, dispatch_command, &kernarg_block,
                    &kernarg_data);
          }
          if (iree_status_is_ok(status)) {
            status = iree_hal_amdgpu_host_queue_replay_dispatch_packet_body(
                block, command_buffer, binding_ptrs, dispatch_command, &packet,
                kernarg_data, iree_hsa_signal_null(), &setup);
          }
          if (iree_status_is_ok(status)) ++packet_count;
        }
        if (iree_status_is_ok(status)) {
          kernarg_block_count +=
              iree_hal_amdgpu_host_queue_dispatch_kernarg_block_count(
                  dispatch_command);
        }
        break;
      }
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_FILL:
        status = iree_hal_amdgpu_host_queue_replay_fill_packet_body(
            queue, command_buffer, binding_table,
            (const iree_hal_amdgpu_command_buffer_fill_command_t*)command,
            &packet, &kernarg_block, iree_hsa_signal_null(), &setup);
        if (iree_status_is_ok(status)) {
          ++packet_count;
          ++kernarg_block_count;
        }
        break;
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COPY:
        status = iree_hal_amdgpu_host_queue_replay_copy_packet_body(
            queue, command_buffer, binding_table,
            (const iree_hal_amdgpu_command_buffer_copy_command_t*)command,
            &packet, &kernarg_block, iree_hsa_signal_null(), &setup);
        if (iree_status_is_ok(status)) {
          ++packet_count;
          ++kernarg_block_count;
        }
        break;
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_UPDATE: {
        const iree_hal_amdgpu_command_buffer_update_command_t* update_command =
            (const iree_hal_amdgpu_command_buffer_update_command_t*)command;
        status = iree_hal_amdgpu_host_queue_check_update_packet_command(
            queue, command_buffer, binding_table, update_command, &packet,
            &setup);
        if (iree_status_is_ok(status)) ++packet_count;
        if (iree_status_is_ok(status)) {
          kernarg_block_count +=
              iree_hal_amdgpu_host_queue_update_kernarg_block_count(
                  update_command->length);
        }
        break;
      }
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN:
        reached_terminator = true;
        break;
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH:
        reached_terminator = true;
        break;
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_PROFILE_MARKER:
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COND_BRANCH:
        status = iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "AQL command-buffer opcode %u replay not yet wired",
            command->opcode);
        break;
      default:
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "malformed AQL command-buffer opcode %u",
                                  command->opcode);
        break;
    }
    if (iree_status_is_ok(status) && !reached_terminator) {
      command = iree_hal_amdgpu_command_buffer_command_next_const(command);
    }
  }
  if (iree_status_is_ok(status) && !reached_terminator) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AQL command-buffer block %" PRIu32
                              " has no terminator",
                              block->block_ordinal);
  }
  if (iree_status_is_ok(status) && packet_count != block->aql_packet_count) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AQL command-buffer block %" PRIu32 " validates %" PRIu32
        " packets but declares %" PRIu32,
        block->block_ordinal, packet_count, block->aql_packet_count);
  }
  const uint32_t declared_kernarg_block_count =
      (uint32_t)iree_host_size_ceil_div(
          block->kernarg_length, sizeof(iree_hal_amdgpu_kernarg_block_t));
  if (iree_status_is_ok(status) &&
      kernarg_block_count != declared_kernarg_block_count) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AQL command-buffer block %" PRIu32 " validates %" PRIu32
        " kernarg blocks but declares %" PRIu32 " kernarg bytes",
        block->block_ordinal, kernarg_block_count, block->kernarg_length);
  }
  return status;
}
#endif  // !defined(NDEBUG)

static iree_status_t iree_hal_amdgpu_host_queue_write_command_buffer_block(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    const uint64_t* binding_ptrs, uint64_t first_payload_packet_id,
    uint32_t packet_index_base, iree_hal_amdgpu_kernarg_block_t* kernarg_blocks,
    uint16_t* packet_headers, uint16_t* packet_setups,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t profile_events,
    uint32_t emitted_packet_count, uint32_t profile_counter_set_count,
    uint32_t profile_trace_packet_count,
    iree_hal_amdgpu_profile_dispatch_harvest_source_t*
        profile_harvest_sources) {
  const iree_hsa_fence_scope_t payload_acquire_scope =
      iree_hal_amdgpu_host_queue_command_buffer_block_payload_acquire_scope(
          queue, block);
  const bool use_base_processor = profile_events.event_count == 0 &&
                                  profile_counter_set_count == 0 &&
                                  profile_trace_packet_count == 0;
  if (use_base_processor) {
    iree_hal_amdgpu_aql_block_processor_t processor;
    const iree_hal_amdgpu_aql_block_processor_t processor_params = {
        .transfer_context = queue->transfer_context,
        .command_buffer = command_buffer,
        .bindings =
            {
                .table = binding_table,
                .ptrs = binding_ptrs,
            },
        .packets =
            {
                .ring = &queue->aql_ring,
                .first_id = first_payload_packet_id,
                .index_base = packet_index_base,
                .count = emitted_packet_count,
                .headers = packet_headers,
                .setups = packet_setups,
            },
        .kernargs =
            {
                .blocks = kernarg_blocks,
                .count = (uint32_t)iree_host_size_ceil_div(
                    block->kernarg_length,
                    sizeof(iree_hal_amdgpu_kernarg_block_t)),
            },
        .submission =
            {
                .wait_barrier_count = resolution->barrier_count,
                .inline_acquire_scope = resolution->inline_acquire_scope,
                .signal_release_scope =
                    iree_hal_amdgpu_host_queue_signal_list_release_scope(
                        queue, signal_semaphore_list),
            },
        .payload =
            {
                .acquire_scope = payload_acquire_scope,
            },
        .flags =
            packet_index_base == 0
                ? IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_FINAL_PAYLOAD_PACKET
                : IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_NONE,
    };
    iree_hal_amdgpu_aql_block_processor_initialize(&processor_params,
                                                   &processor);
    iree_hal_amdgpu_aql_block_processor_result_t result;
    iree_status_t status =
        iree_hal_amdgpu_aql_block_processor_invoke(&processor, block, &result);
    iree_hal_amdgpu_aql_block_processor_deinitialize(&processor);
    return status;
  }
  // Per-dispatch counter and trace packets are emitted before the recorded
  // payload packet they wrap. Do not let a submit-time barrier on logical
  // packet 0 shrink the recorded payload acquire span when that first logical
  // packet is profiling metadata instead of the recorded payload stream.
  const uint32_t first_recorded_packet_index_base =
      profile_counter_set_count == 0 && profile_trace_packet_count == 0
          ? packet_index_base
          : 1u;
  const uint32_t payload_acquire_packet_count =
      iree_hal_amdgpu_host_queue_command_buffer_block_payload_acquire_packet_count(
          resolution, block, first_recorded_packet_index_base,
          payload_acquire_scope);
  iree_hal_amdgpu_aql_block_processor_profile_flags_t profile_flags =
      IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PROFILE_FLAG_NONE;
  if (profile_events.event_count != 0) {
    profile_flags |=
        IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PROFILE_FLAG_DISPATCH_PACKETS;
  }
  if (packet_index_base != 0) {
    profile_flags |=
        IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PROFILE_FLAG_QUEUE_DEVICE_EVENT;
  }
  iree_hal_amdgpu_aql_block_processor_profile_t processor;
  const iree_hal_amdgpu_aql_block_processor_profile_t processor_params = {
      .queue = queue,
      .command_buffer = command_buffer,
      .block = block,
      .submission =
          {
              .resolution = resolution,
              .signal_semaphore_list = signal_semaphore_list,
          },
      .bindings =
          {
              .table = binding_table,
              .ptrs = binding_ptrs,
          },
      .packets =
          {
              .first_payload_id = first_payload_packet_id,
              .index_base = packet_index_base,
              .count = emitted_packet_count,
              .headers = packet_headers,
              .setups = packet_setups,
          },
      .kernargs =
          {
              .blocks = kernarg_blocks,
              .count = (uint32_t)iree_host_size_ceil_div(
                  block->kernarg_length,
                  sizeof(iree_hal_amdgpu_kernarg_block_t)),
          },
      .payload =
          {
              .acquire_scope = payload_acquire_scope,
              .acquire_packet_count = payload_acquire_packet_count,
          },
      .profile =
          {
              .dispatch_events = profile_events,
              .harvest_sources = profile_harvest_sources,
              .command_buffer_id =
                  iree_hal_amdgpu_aql_command_buffer_profile_id(command_buffer),
              .counter_set_count = profile_counter_set_count,
              .trace_packet_count = profile_trace_packet_count,
          },
      .flags = profile_flags,
  };
  iree_hal_amdgpu_aql_block_processor_profile_initialize(&processor_params,
                                                         &processor);
  iree_hal_amdgpu_aql_block_processor_profile_result_t result;
  iree_status_t status =
      iree_hal_amdgpu_aql_block_processor_profile_invoke(&processor, &result);
  iree_hal_amdgpu_aql_block_processor_profile_deinitialize(&processor);
  return status;
}

typedef uint32_t
    iree_hal_amdgpu_host_queue_command_buffer_profile_submission_flags_t;
enum iree_hal_amdgpu_host_queue_command_buffer_profile_submission_flag_bits_t {
  IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PROFILE_SUBMISSION_FLAG_NONE = 0u,
  // The non-profiling path needs a trailing barrier packet to own queue
  // completion.
  IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PROFILE_SUBMISSION_FLAG_TRAILING_COMPLETION_PACKET =
      1u << 0,
};

typedef struct iree_hal_amdgpu_host_queue_command_buffer_profile_submission_t {
  // Reserved dispatch timestamp events for profiled commands in this block.
  iree_hal_amdgpu_profile_dispatch_event_reservation_t dispatch_events;
  // Reserved whole-block queue-device timestamp event for this execute.
  iree_hal_amdgpu_profile_queue_device_event_reservation_t queue_device_events;
  // Host queue event metadata shared by host and device queue-event records.
  iree_hal_amdgpu_host_queue_profile_event_info_t queue_event_info;
  // Optional harvest dispatch emitted after profiled dispatch payloads.
  struct {
    // Dispatch packet for harvesting dispatch timestamp records.
    iree_hal_amdgpu_aql_packet_t* packet;
    // Setup bits for |packet| when present.
    uint16_t setup;
  } harvest;
  // Flags from
  // iree_hal_amdgpu_host_queue_command_buffer_profile_submission_flag_bits_t.
  iree_hal_amdgpu_host_queue_command_buffer_profile_submission_flags_t flags;
} iree_hal_amdgpu_host_queue_command_buffer_profile_submission_t;

// Publishes the non-profiling terminal barrier packet for a replayed
// command-buffer block. Payload packets keep their recorded final-payload
// barriers, but queue completion is signaled from this trailing packet so
// software observes block completion only after the CP reaches the end of the
// replay span.
static void iree_hal_amdgpu_host_queue_commit_command_buffer_completion_packet(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list, uint64_t packet_id,
    uint32_t packet_index) {
  iree_hal_amdgpu_aql_packet_t* packet =
      iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, packet_id);
  const uint16_t header = iree_hal_amdgpu_aql_emit_nop(
      &packet->barrier_and,
      iree_hal_amdgpu_host_queue_command_buffer_packet_control(
          queue, resolution, signal_semaphore_list, packet_index,
          IREE_HSA_FENCE_SCOPE_NONE,
          IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_FINAL),
      iree_hal_amdgpu_notification_ring_epoch_signal(
          &queue->notification_ring));
  iree_hal_amdgpu_aql_ring_commit(packet, header, /*setup=*/0);
}

static uint64_t iree_hal_amdgpu_host_queue_finish_command_buffer_block(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    uint32_t emitted_packet_count,
    iree_hal_resource_set_t** inout_binding_resource_set,
    iree_hal_amdgpu_reclaim_action_t pre_signal_action,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_host_queue_kernel_submission_t* submission,
    const uint16_t* packet_headers, const uint16_t* packet_setups,
    iree_hal_amdgpu_host_queue_command_buffer_profile_submission_t* profile) {
  submission->pre_signal_action = pre_signal_action;
  iree_hal_amdgpu_host_queue_emit_kernel_submission_prefix(queue, resolution,
                                                           submission);
  const uint64_t submission_epoch =
      iree_hal_amdgpu_host_queue_finish_kernel_submission(
          queue, resolution, signal_semaphore_list, operation_resources,
          operation_resource_count, inout_binding_resource_set,
          submission_flags, submission);

  iree_hal_amdgpu_profile_queue_device_event_t* queue_device_event =
      iree_hal_amdgpu_host_queue_initialize_profile_queue_device_event(
          queue, profile->queue_device_events, &profile->queue_event_info);
  if (queue_device_event) {
    submission->reclaim_entry->queue_device_event_first_position =
        profile->queue_device_events.first_event_position;
    submission->reclaim_entry->queue_device_event_count =
        profile->queue_device_events.event_count;
    queue_device_event->submission_id = submission_epoch;
  }

  uint16_t profile_harvest_header = 0;
  if (profile->dispatch_events.event_count != 0) {
    submission->reclaim_entry->profile_event_first_position =
        profile->dispatch_events.first_event_position;
    submission->reclaim_entry->profile_event_count =
        profile->dispatch_events.event_count;
    for (uint32_t i = 0; i < profile->dispatch_events.event_count; ++i) {
      iree_hal_amdgpu_profile_dispatch_event_t* event =
          iree_hal_amdgpu_host_queue_profile_dispatch_event_at(
              queue, profile->dispatch_events.first_event_position + i);
      event->submission_id = submission_epoch;
    }
    profile->harvest.packet->dispatch.completion_signal =
        queue_device_event ? iree_hsa_signal_null()
                           : iree_hal_amdgpu_notification_ring_epoch_signal(
                                 &queue->notification_ring);
    const iree_hsa_fence_scope_t profile_harvest_acquire_scope =
        iree_hal_amdgpu_host_queue_kernarg_acquire_scope(
            queue, IREE_HSA_FENCE_SCOPE_AGENT);
    profile_harvest_header = iree_hal_amdgpu_aql_make_header(
        IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
        iree_hal_amdgpu_aql_packet_control_barrier(
            iree_hal_amdgpu_host_queue_max_fence_scope(
                profile_harvest_acquire_scope,
                resolution->inline_acquire_scope),
            IREE_HSA_FENCE_SCOPE_SYSTEM));
  }

  const uint32_t profile_queue_device_prefix_packet_count =
      queue_device_event ? 1u : 0u;
  const uint64_t first_payload_packet_id =
      submission->first_packet_id + resolution->barrier_count +
      profile_queue_device_prefix_packet_count;
  iree_hal_amdgpu_host_queue_publish_submission_kernargs(queue, submission);
  if (queue_device_event) {
    const uint64_t start_packet_id =
        submission->first_packet_id + resolution->barrier_count;
    iree_hal_amdgpu_host_queue_commit_command_buffer_profile_start(
        queue, start_packet_id,
        iree_hal_amdgpu_host_queue_command_buffer_packet_control(
            queue, resolution, signal_semaphore_list, /*packet_index=*/0,
            IREE_HSA_FENCE_SCOPE_NONE,
            IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_NONE),
        queue_device_event);
  }

  for (uint32_t i = 0; i < emitted_packet_count; ++i) {
    iree_hal_amdgpu_aql_packet_t* packet = iree_hal_amdgpu_aql_ring_packet(
        &queue->aql_ring, first_payload_packet_id + i);
    if (iree_hal_amdgpu_host_queue_aql_packet_header_type(packet_headers[i]) !=
        IREE_HSA_PACKET_TYPE_INVALID) {
      iree_hal_amdgpu_aql_ring_commit(packet, packet_headers[i],
                                      packet_setups[i]);
    }
  }
  if (profile->dispatch_events.event_count != 0) {
    iree_hal_amdgpu_aql_ring_commit(profile->harvest.packet,
                                    profile_harvest_header,
                                    profile->harvest.setup);
  }
  if (iree_any_bit_set(
          profile->flags,
          IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PROFILE_SUBMISSION_FLAG_TRAILING_COMPLETION_PACKET)) {
    const uint32_t profile_harvest_packet_count =
        profile->dispatch_events.event_count != 0 ? 1u : 0u;
    const uint64_t completion_packet_id = first_payload_packet_id +
                                          emitted_packet_count +
                                          profile_harvest_packet_count;
    const uint32_t completion_packet_index =
        profile_queue_device_prefix_packet_count + emitted_packet_count +
        profile_harvest_packet_count;
    iree_hal_amdgpu_host_queue_commit_command_buffer_completion_packet(
        queue, resolution, signal_semaphore_list, completion_packet_id,
        completion_packet_index);
  }
  if (queue_device_event) {
    const uint64_t end_packet_id =
        first_payload_packet_id + emitted_packet_count +
        (profile->dispatch_events.event_count != 0 ? 1u : 0u);
    const uint32_t end_packet_index =
        profile_queue_device_prefix_packet_count + emitted_packet_count +
        (profile->dispatch_events.event_count != 0 ? 1u : 0u);
    iree_hal_amdgpu_host_queue_commit_command_buffer_profile_end(
        queue, end_packet_id,
        iree_hal_amdgpu_host_queue_command_buffer_packet_control(
            queue, resolution, signal_semaphore_list, end_packet_index,
            IREE_HSA_FENCE_SCOPE_NONE,
            IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_FINAL),
        iree_hal_amdgpu_notification_ring_epoch_signal(
            &queue->notification_ring),
        queue_device_event);
  }
  iree_hal_amdgpu_aql_ring_doorbell(
      &queue->aql_ring,
      submission->first_packet_id + submission->packet_count - 1);
  profile->queue_event_info.submission_id = submission_epoch;
  memset(submission, 0, sizeof(*submission));
  return submission_epoch;
}

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
    bool* out_ready) {
  *out_ready = false;
  const uint64_t command_buffer_id =
      iree_hal_amdgpu_aql_command_buffer_profile_id(command_buffer);
  const uint32_t kernarg_block_count = (uint32_t)iree_host_size_ceil_div(
      block->kernarg_length, sizeof(iree_hal_amdgpu_kernarg_block_t));
  uint32_t profile_dispatch_event_count = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_count_command_buffer_profile_dispatch_events(
          queue, command_buffer, block, &profile_dispatch_event_count));
  iree_hal_amdgpu_profile_dispatch_event_reservation_t profile_events = {0};
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_reserve_profile_dispatch_events(
          queue, profile_dispatch_event_count, &profile_events));
  const uint32_t profile_counter_set_count =
      profile_events.event_count != 0
          ? iree_hal_amdgpu_host_queue_profile_counter_set_count(queue,
                                                                 profile_events)
          : 0u;
  const uint32_t profile_counter_packet_count =
      profile_events.event_count != 0
          ? iree_hal_amdgpu_host_queue_profile_counter_packet_count(
                queue, profile_events)
          : 0u;
  const uint32_t profile_trace_packet_count =
      profile_events.event_count != 0
          ? iree_hal_amdgpu_host_queue_profile_trace_packet_count(
                queue, profile_events)
          : 0u;
  if (IREE_UNLIKELY(profile_trace_packet_count >
                    UINT32_MAX - profile_counter_packet_count)) {
    iree_hal_amdgpu_host_queue_cancel_profile_dispatch_events(queue,
                                                              profile_events);
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "profiled command-buffer block packet count overflow");
  }
  const uint32_t extra_profile_packet_count =
      profile_counter_packet_count + profile_trace_packet_count;
  if (IREE_UNLIKELY(block->aql_packet_count >
                    UINT32_MAX - extra_profile_packet_count)) {
    iree_hal_amdgpu_host_queue_cancel_profile_dispatch_events(queue,
                                                              profile_events);
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "profiled command-buffer block packet count overflow");
  }
  const uint32_t emitted_packet_count =
      block->aql_packet_count + extra_profile_packet_count;
  iree_hal_amdgpu_profile_queue_device_event_reservation_t
      profile_queue_device_events = {0};
  if (iree_hal_amdgpu_host_queue_should_profile_queue_device_events(queue)) {
    iree_status_t reserve_status =
        iree_hal_amdgpu_host_queue_reserve_profile_queue_device_events(
            queue, /*event_count=*/1, &profile_queue_device_events);
    if (!iree_status_is_ok(reserve_status)) {
      iree_hal_amdgpu_host_queue_cancel_profile_dispatch_events(queue,
                                                                profile_events);
      return reserve_status;
    }
  }
  const uint32_t profile_harvest_packet_count =
      profile_events.event_count != 0 ? 1u : 0u;
  const uint32_t profile_queue_device_packet_count =
      profile_queue_device_events.event_count != 0 ? 2u : 0u;
  iree_hal_amdgpu_host_queue_command_buffer_profile_submission_flags_t
      profile_submission_flags =
          IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PROFILE_SUBMISSION_FLAG_NONE;
  if (profile_events.event_count == 0 &&
      profile_queue_device_packet_count == 0) {
    profile_submission_flags |=
        IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PROFILE_SUBMISSION_FLAG_TRAILING_COMPLETION_PACKET;
  }
  const uint32_t trailing_completion_packet_count =
      iree_any_bit_set(
          profile_submission_flags,
          IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PROFILE_SUBMISSION_FLAG_TRAILING_COMPLETION_PACKET)
          ? 1u
          : 0u;
  if (IREE_UNLIKELY(emitted_packet_count >
                        UINT32_MAX - profile_harvest_packet_count ||
                    emitted_packet_count + profile_harvest_packet_count >
                        UINT32_MAX - profile_queue_device_packet_count ||
                    emitted_packet_count + profile_harvest_packet_count +
                            profile_queue_device_packet_count >
                        UINT32_MAX - trailing_completion_packet_count)) {
    iree_hal_amdgpu_host_queue_cancel_profile_dispatch_events(queue,
                                                              profile_events);
    iree_hal_amdgpu_host_queue_cancel_profile_queue_device_events(
        queue, profile_queue_device_events);
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "profiled command-buffer block packet count overflow");
  }
  const uint32_t payload_packet_count =
      emitted_packet_count + profile_harvest_packet_count +
      profile_queue_device_packet_count + trailing_completion_packet_count;
  const uint32_t profile_harvest_kernarg_block_count =
      profile_events.event_count != 0
          ? (uint32_t)iree_host_size_ceil_div(
                iree_hal_amdgpu_device_profile_dispatch_harvest_kernarg_length(
                    profile_events.event_count),
                sizeof(iree_hal_amdgpu_kernarg_block_t))
          : 0u;

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(queue->block_pool, &scratch_arena);
  const uint64_t* binding_ptrs = NULL;
  iree_status_t status =
      iree_hal_amdgpu_host_queue_prepare_command_buffer_binding_ptrs(
          queue, command_buffer, binding_table, block, &scratch_arena,
          &binding_ptrs);
#if !defined(NDEBUG)
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_check_packet_commands(
        queue, command_buffer, binding_table, binding_ptrs, &scratch_arena,
        block);
  }
#endif  // !defined(NDEBUG)
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_prepare_profile_counter_samples(
        queue, profile_events);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_prepare_profile_traces(queue,
                                                               profile_events);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_amdgpu_host_queue_prepare_command_buffer_profile_trace_code_objects(
            queue, command_buffer, block, profile_events);
  }

  iree_hal_amdgpu_host_queue_kernel_submission_t submission;
  memset(&submission, 0, sizeof(submission));
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_try_begin_kernel_submission(
        queue, resolution, signal_semaphore_list, operation_resource_count,
        payload_packet_count,
        kernarg_block_count + profile_harvest_kernarg_block_count, out_ready,
        &submission);
  }
  if (!iree_status_is_ok(status) || !*out_ready) {
    iree_hal_amdgpu_host_queue_cancel_profile_dispatch_events(queue,
                                                              profile_events);
    iree_hal_amdgpu_host_queue_cancel_profile_queue_device_events(
        queue, profile_queue_device_events);
  }
  if (iree_status_is_ok(status) && *out_ready) {
    iree_hal_amdgpu_aql_packet_t* profile_harvest_packet = NULL;
    iree_hal_amdgpu_profile_dispatch_harvest_source_t* profile_harvest_sources =
        NULL;
    uint16_t profile_harvest_setup = 0;
    const uint32_t profile_queue_device_prefix_packet_count =
        profile_queue_device_events.event_count != 0 ? 1u : 0u;
    const uint64_t first_payload_packet_id =
        submission.first_packet_id + resolution->barrier_count +
        profile_queue_device_prefix_packet_count;
    if (profile_events.event_count != 0) {
      profile_harvest_packet = iree_hal_amdgpu_aql_ring_packet(
          &queue->aql_ring, first_payload_packet_id + emitted_packet_count);
      profile_harvest_sources =
          iree_hal_amdgpu_device_profile_emplace_dispatch_harvest(
              &queue->transfer_context->kernels
                   ->iree_hal_amdgpu_device_profile_harvest_dispatch_events,
              profile_events.event_count, &profile_harvest_packet->dispatch,
              submission.kernarg_blocks[kernarg_block_count].data);
      profile_harvest_setup = profile_harvest_packet->dispatch.setup;
    }
    uint16_t* packet_headers = NULL;
    uint16_t* packet_setups = NULL;
    status = iree_hal_amdgpu_host_queue_prepare_command_buffer_packet_metadata(
        queue, emitted_packet_count, &scratch_arena, &packet_headers,
        &packet_setups);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_host_queue_write_command_buffer_block(
          queue, resolution, signal_semaphore_list, command_buffer,
          binding_table, block, binding_ptrs, first_payload_packet_id,
          profile_queue_device_prefix_packet_count, submission.kernarg_blocks,
          packet_headers, packet_setups, profile_events, emitted_packet_count,
          profile_counter_set_count, profile_trace_packet_count,
          profile_harvest_sources);
    }
    if (iree_status_is_ok(status)) {
      iree_hal_amdgpu_host_queue_command_buffer_profile_submission_t
          profile_submission = {
              .dispatch_events = profile_events,
              .queue_device_events = profile_queue_device_events,
              .queue_event_info =
                  {
                      .type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE,
                      .command_buffer_id = command_buffer_id,
                      .operation_count = block->command_count,
                  },
              .harvest =
                  {
                      .packet = profile_harvest_packet,
                      .setup = profile_harvest_setup,
                  },
              .flags = profile_submission_flags,
          };
      iree_hal_amdgpu_host_queue_finish_command_buffer_block(
          queue, resolution, signal_semaphore_list, emitted_packet_count,
          inout_binding_resource_set, pre_signal_action, operation_resources,
          operation_resource_count, submission_flags, &submission,
          packet_headers, packet_setups, &profile_submission);
      iree_hal_amdgpu_host_queue_record_profile_queue_event(
          queue, resolution, signal_semaphore_list,
          &profile_submission.queue_event_info);
    } else {
      iree_hal_amdgpu_host_queue_fail_kernel_submission(queue, &submission);
      iree_hal_amdgpu_host_queue_cancel_profile_dispatch_events(queue,
                                                                profile_events);
      iree_hal_amdgpu_host_queue_cancel_profile_queue_device_events(
          queue, profile_queue_device_events);
    }
  }
  iree_arena_deinitialize(&scratch_arena);
  return status;
}

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/aql_block_processor.h"

#include <cstring>

#include "iree/hal/drivers/amdgpu/abi/queue.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

struct ReturnBlock {
  // Block header at the ABI-defined block base.
  iree_hal_amdgpu_command_buffer_block_header_t header;
  // Single return terminator command.
  iree_hal_amdgpu_command_buffer_return_command_t return_command;
};

struct BranchBlock {
  // Block header at the ABI-defined block base.
  iree_hal_amdgpu_command_buffer_block_header_t header;
  // Single branch terminator command.
  iree_hal_amdgpu_command_buffer_branch_command_t branch_command;
};

struct DirectDispatchBlock {
  // Block header at the ABI-defined block base.
  iree_hal_amdgpu_command_buffer_block_header_t header;
  // Custom-direct dispatch command under test.
  iree_hal_amdgpu_command_buffer_dispatch_command_t dispatch_command;
  // Inline custom-direct kernarg tail copied into queue-owned kernargs.
  uint64_t tail[2];
  // Return terminator following the dispatch command and inline tail.
  iree_hal_amdgpu_command_buffer_return_command_t return_command;
};

template <uint32_t DispatchCount>
struct DispatchBlock {
  // Block header at the ABI-defined block base.
  iree_hal_amdgpu_command_buffer_block_header_t header;
  // Direct dispatch commands recorded in this block.
  iree_hal_amdgpu_command_buffer_dispatch_command_t
      dispatch_commands[DispatchCount];
  // Return terminator following the dispatch commands.
  iree_hal_amdgpu_command_buffer_return_command_t return_command;
};

struct MalformedBlock {
  // Block header at the ABI-defined block base.
  iree_hal_amdgpu_command_buffer_block_header_t header;
  // Non-terminating barrier command used to exercise validation.
  iree_hal_amdgpu_command_buffer_barrier_command_t barrier_command;
};

struct PacketHeaderSummary {
  // Counts accumulated across emitted packet headers.
  struct {
    // Number of emitted packet headers summarized.
    uint32_t total;
    // Number of emitted packet headers carrying the AQL barrier bit.
    uint32_t barrier;
    // Number of emitted packet headers with SYSTEM acquire scope.
    uint32_t system_acquire;
    // Number of emitted packet headers with SYSTEM release scope.
    uint32_t system_release;
  } counts;
  // Boundary packet headers from the emitted span.
  struct {
    // First emitted packet header, or zero for empty spans.
    uint16_t first;
    // Last emitted packet header, or zero for empty spans.
    uint16_t last;
  } headers;
};

static void InitializeBlockHeader(
    uint32_t block_length, uint32_t command_length, uint16_t command_count,
    uint32_t aql_packet_count, uint32_t kernarg_length,
    iree_hal_amdgpu_command_buffer_block_header_t* out_header) {
  std::memset(out_header, 0, sizeof(*out_header));
  out_header->magic = IREE_HAL_AMDGPU_COMMAND_BUFFER_BLOCK_MAGIC;
  out_header->version = IREE_HAL_AMDGPU_COMMAND_BUFFER_BLOCK_VERSION_0;
  out_header->header_length = sizeof(*out_header);
  out_header->block_length = block_length;
  out_header->command_offset = sizeof(*out_header);
  out_header->command_length = command_length;
  out_header->command_count = command_count;
  out_header->aql_packet_count = aql_packet_count;
  out_header->kernarg_length = kernarg_length;
  out_header->initial_barrier_packet_count = aql_packet_count;
  out_header->binding_source_offset = block_length;
  out_header->rodata_offset = block_length;
}

static void InitializeReturnCommand(
    iree_hal_amdgpu_command_buffer_return_command_t* out_command) {
  std::memset(out_command, 0, sizeof(*out_command));
  out_command->header.opcode = IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN;
  out_command->header.length_qwords =
      sizeof(*out_command) / IREE_HAL_AMDGPU_COMMAND_BUFFER_RECORD_ALIGNMENT;
}

static void SetReturnTerminator(
    iree_hal_amdgpu_command_buffer_block_header_t* block_header) {
  block_header->terminator_opcode =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN;
  block_header->terminator_target_block_ordinal = 0;
}

static void SetBranchTerminator(
    uint32_t target_block_ordinal,
    iree_hal_amdgpu_command_buffer_block_header_t* block_header) {
  block_header->terminator_opcode =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH;
  block_header->terminator_target_block_ordinal = target_block_ordinal;
}

static uint8_t CommandFlags(uint8_t flags, iree_hsa_fence_scope_t acquire_scope,
                            iree_hsa_fence_scope_t release_scope) {
  return iree_hal_amdgpu_command_buffer_command_flags_set_fence_scopes(
      flags, (uint8_t)acquire_scope, (uint8_t)release_scope);
}

static void InitializeDirectDispatchCommand(
    uint32_t command_index, uint8_t command_flags,
    iree_hal_amdgpu_command_buffer_dispatch_command_t* out_command) {
  std::memset(out_command, 0, sizeof(*out_command));
  out_command->header.opcode = IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH;
  out_command->header.flags = command_flags;
  out_command->header.length_qwords =
      sizeof(*out_command) / IREE_HAL_AMDGPU_COMMAND_BUFFER_RECORD_ALIGNMENT;
  out_command->header.command_index = command_index;
  out_command->kernel_object = 0xABCDEF0000000000ull + command_index;
  out_command->payload_reference = sizeof(*out_command);
  out_command->kernarg_strategy =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_CUSTOM_DIRECT;
  out_command->setup = 3;
  out_command->workgroup_size[0] = 1;
  out_command->workgroup_size[1] = 1;
  out_command->workgroup_size[2] = 1;
  out_command->grid_size[0] = 1;
  out_command->grid_size[1] = 1;
  out_command->grid_size[2] = 1;
}

static ReturnBlock MakeReturnBlock() {
  ReturnBlock block;
  InitializeBlockHeader(sizeof(block), sizeof(block.return_command),
                        /*command_count=*/1, /*aql_packet_count=*/0,
                        /*kernarg_length=*/0, &block.header);
  InitializeReturnCommand(&block.return_command);
  SetReturnTerminator(&block.header);
  return block;
}

static BranchBlock MakeBranchBlock(uint32_t target_block_ordinal) {
  BranchBlock block;
  InitializeBlockHeader(sizeof(block), sizeof(block.branch_command),
                        /*command_count=*/1, /*aql_packet_count=*/0,
                        /*kernarg_length=*/0, &block.header);
  std::memset(&block.branch_command, 0, sizeof(block.branch_command));
  block.branch_command.header.opcode =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH;
  block.branch_command.header.length_qwords =
      sizeof(block.branch_command) /
      IREE_HAL_AMDGPU_COMMAND_BUFFER_RECORD_ALIGNMENT;
  block.branch_command.target_block_ordinal = target_block_ordinal;
  SetBranchTerminator(target_block_ordinal, &block.header);
  return block;
}

static MalformedBlock MakeUnterminatedBlock() {
  MalformedBlock block;
  InitializeBlockHeader(sizeof(block), sizeof(block.barrier_command),
                        /*command_count=*/1, /*aql_packet_count=*/0,
                        /*kernarg_length=*/0, &block.header);
  std::memset(&block.barrier_command, 0, sizeof(block.barrier_command));
  block.barrier_command.header.opcode =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BARRIER;
  block.barrier_command.header.length_qwords =
      sizeof(block.barrier_command) /
      IREE_HAL_AMDGPU_COMMAND_BUFFER_RECORD_ALIGNMENT;
  return block;
}

static DirectDispatchBlock MakeDirectDispatchBlock() {
  DirectDispatchBlock block;
  const uint32_t dispatch_command_length =
      sizeof(block.dispatch_command) + sizeof(block.tail);
  const uint32_t command_length =
      dispatch_command_length + sizeof(block.return_command);
  InitializeBlockHeader(sizeof(block), command_length, /*command_count=*/2,
                        /*aql_packet_count=*/1,
                        /*kernarg_length=*/sizeof(block.tail), &block.header);

  std::memset(&block.dispatch_command, 0, sizeof(block.dispatch_command));
  block.dispatch_command.header.opcode =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH;
  block.dispatch_command.header.flags =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS;
  block.dispatch_command.header.length_qwords =
      dispatch_command_length / IREE_HAL_AMDGPU_COMMAND_BUFFER_RECORD_ALIGNMENT;
  block.dispatch_command.kernel_object = 0x123456789ABCDEF0ull;
  block.dispatch_command.payload_reference = sizeof(block.dispatch_command);
  block.dispatch_command.kernarg_length_qwords =
      sizeof(block.tail) / sizeof(uint64_t);
  block.dispatch_command.tail_length_qwords =
      sizeof(block.tail) / sizeof(uint64_t);
  block.dispatch_command.kernarg_strategy =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_CUSTOM_DIRECT;
  block.dispatch_command.setup = 3;
  block.dispatch_command.workgroup_size[0] = 4;
  block.dispatch_command.workgroup_size[1] = 2;
  block.dispatch_command.workgroup_size[2] = 1;
  block.dispatch_command.grid_size[0] = 64;
  block.dispatch_command.grid_size[1] = 8;
  block.dispatch_command.grid_size[2] = 1;
  block.dispatch_command.private_segment_size = 128;
  block.dispatch_command.group_segment_size = 256;
  block.tail[0] = 0x0A0B0C0D0E0F1011ull;
  block.tail[1] = 0x1213141516171819ull;

  block.header.dispatch_count = 1;
  InitializeReturnCommand(&block.return_command);
  SetReturnTerminator(&block.header);
  return block;
}

template <uint32_t DispatchCount>
static DispatchBlock<DispatchCount> MakeDispatchBlock(
    const uint8_t (&dispatch_command_flags)[DispatchCount]) {
  DispatchBlock<DispatchCount> block;
  const uint32_t command_length =
      DispatchCount * sizeof(block.dispatch_commands[0]) +
      sizeof(block.return_command);
  InitializeBlockHeader(sizeof(block), command_length,
                        /*command_count=*/DispatchCount + 1,
                        /*aql_packet_count=*/DispatchCount,
                        /*kernarg_length=*/DispatchCount *
                            sizeof(iree_hal_amdgpu_kernarg_block_t),
                        &block.header);
  block.header.dispatch_count = DispatchCount;
  for (uint32_t i = 0; i < DispatchCount; ++i) {
    InitializeDirectDispatchCommand(i, dispatch_command_flags[i],
                                    &block.dispatch_commands[i]);
  }
  InitializeReturnCommand(&block.return_command);
  SetReturnTerminator(&block.header);
  return block;
}

static iree_hal_amdgpu_aql_block_processor_t MakeProcessor(
    iree_hal_amdgpu_aql_ring_t* ring, uint32_t packet_count,
    uint16_t* packet_headers, uint16_t* packet_setups,
    iree_hal_amdgpu_kernarg_block_t* kernarg_blocks,
    uint32_t kernarg_block_count,
    iree_hal_amdgpu_aql_block_processor_flags_t flags =
        IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_NONE,
    iree_hsa_fence_scope_t inline_acquire_scope = IREE_HSA_FENCE_SCOPE_NONE,
    iree_hsa_fence_scope_t signal_release_scope = IREE_HSA_FENCE_SCOPE_SYSTEM,
    iree_hsa_fence_scope_t payload_acquire_scope =
        IREE_HSA_FENCE_SCOPE_SYSTEM) {
  iree_hal_amdgpu_aql_block_processor_t processor = {};
  processor.packets.ring = ring;
  processor.packets.first_id = 4;
  processor.packets.index_base = 0;
  processor.packets.count = packet_count;
  processor.packets.headers = packet_headers;
  processor.packets.setups = packet_setups;
  processor.kernargs.blocks = kernarg_blocks;
  processor.kernargs.count = kernarg_block_count;
  processor.submission.inline_acquire_scope = inline_acquire_scope;
  processor.submission.signal_release_scope = signal_release_scope;
  processor.payload.acquire_scope = payload_acquire_scope;
  processor.flags = flags;
  return processor;
}

static uint16_t AqlHeaderField(uint16_t header, uint32_t bit_offset,
                               uint32_t bit_width) {
  return (header >> bit_offset) & ((1u << bit_width) - 1u);
}

static bool AqlHeaderHasBarrier(uint16_t header) {
  return AqlHeaderField(header, IREE_HSA_PACKET_HEADER_BARRIER,
                        IREE_HSA_PACKET_HEADER_WIDTH_BARRIER) != 0;
}

static iree_hsa_fence_scope_t AqlHeaderAcquireScope(uint16_t header) {
  return (iree_hsa_fence_scope_t)AqlHeaderField(
      header, IREE_HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE,
      IREE_HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE);
}

static iree_hsa_fence_scope_t AqlHeaderReleaseScope(uint16_t header) {
  return (iree_hsa_fence_scope_t)AqlHeaderField(
      header, IREE_HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE,
      IREE_HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE);
}

static PacketHeaderSummary SummarizePacketHeaders(
    const uint16_t* packet_headers, uint32_t packet_count) {
  PacketHeaderSummary summary = {};
  for (uint32_t i = 0; i < packet_count; ++i) {
    const uint16_t header = packet_headers[i];
    if (summary.counts.total == 0) summary.headers.first = header;
    summary.headers.last = header;
    ++summary.counts.total;
    if (AqlHeaderHasBarrier(header)) ++summary.counts.barrier;
    if (AqlHeaderAcquireScope(header) == IREE_HSA_FENCE_SCOPE_SYSTEM) {
      ++summary.counts.system_acquire;
    }
    if (AqlHeaderReleaseScope(header) == IREE_HSA_FENCE_SCOPE_SYSTEM) {
      ++summary.counts.system_release;
    }
  }
  return summary;
}

template <uint32_t DispatchCount>
static iree_status_t InvokeAndSummarizeDispatchBlock(
    const DispatchBlock<DispatchCount>& block,
    iree_hal_amdgpu_aql_block_processor_flags_t flags,
    iree_hsa_fence_scope_t inline_acquire_scope,
    iree_hsa_fence_scope_t signal_release_scope,
    iree_hsa_fence_scope_t payload_acquire_scope,
    PacketHeaderSummary* out_summary) {
  alignas(64) iree_hal_amdgpu_aql_packet_t packets[8] = {};
  iree_hal_amdgpu_aql_ring_t ring = {};
  ring.base = packets;
  ring.mask = IREE_ARRAYSIZE(packets) - 1u;
  uint16_t packet_headers[DispatchCount] = {};
  uint16_t packet_setups[DispatchCount] = {};
  iree_hal_amdgpu_kernarg_block_t kernarg_blocks[DispatchCount] = {};
  iree_hal_amdgpu_aql_block_processor_t processor = MakeProcessor(
      &ring, /*packet_count=*/DispatchCount, packet_headers, packet_setups,
      kernarg_blocks, /*kernarg_block_count=*/DispatchCount, flags,
      inline_acquire_scope, signal_release_scope, payload_acquire_scope);

  iree_hal_amdgpu_aql_block_processor_result_t result;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_block_processor_invoke(
      &processor, &block.header, &result));
  *out_summary = SummarizePacketHeaders(packet_headers, DispatchCount);
  return iree_ok_status();
}

TEST(AqlBlockProcessorTest, ReturnTerminatorProducesNoPayload) {
  ReturnBlock block = MakeReturnBlock();
  iree_hal_amdgpu_aql_block_processor_t processor =
      MakeProcessor(/*ring=*/nullptr, /*packet_count=*/0,
                    /*packet_headers=*/nullptr, /*packet_setups=*/nullptr,
                    /*kernarg_blocks=*/nullptr, /*kernarg_block_count=*/0);

  iree_hal_amdgpu_aql_block_processor_result_t result;
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_block_processor_invoke(
      &processor, &block.header, &result));

  EXPECT_EQ(result.terminator,
            IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_TERMINATOR_RETURN);
  EXPECT_EQ(block.header.terminator_opcode,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN);
  EXPECT_EQ(result.packets.recorded, 0u);
  EXPECT_EQ(result.packets.emitted, 0u);
  EXPECT_EQ(result.kernargs.consumed, 0u);
}

TEST(AqlBlockProcessorTest, BranchTerminatorReportsTargetBlock) {
  BranchBlock block = MakeBranchBlock(/*target_block_ordinal=*/7);
  iree_hal_amdgpu_aql_block_processor_t processor =
      MakeProcessor(/*ring=*/nullptr, /*packet_count=*/0,
                    /*packet_headers=*/nullptr, /*packet_setups=*/nullptr,
                    /*kernarg_blocks=*/nullptr, /*kernarg_block_count=*/0);

  iree_hal_amdgpu_aql_block_processor_result_t result;
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_block_processor_invoke(
      &processor, &block.header, &result));

  EXPECT_EQ(result.terminator,
            IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_TERMINATOR_BRANCH);
  EXPECT_EQ(result.target_block_ordinal, 7u);
  EXPECT_EQ(block.header.terminator_opcode,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH);
  EXPECT_EQ(block.header.terminator_target_block_ordinal,
            result.target_block_ordinal);
  EXPECT_EQ(result.packets.recorded, 0u);
  EXPECT_EQ(result.packets.emitted, 0u);
  EXPECT_EQ(result.kernargs.consumed, 0u);
}

TEST(AqlBlockProcessorTest, UnterminatedBlockFails) {
  MalformedBlock block = MakeUnterminatedBlock();
  iree_hal_amdgpu_aql_block_processor_t processor =
      MakeProcessor(/*ring=*/nullptr, /*packet_count=*/0,
                    /*packet_headers=*/nullptr, /*packet_setups=*/nullptr,
                    /*kernarg_blocks=*/nullptr, /*kernarg_block_count=*/0);

  iree_hal_amdgpu_aql_block_processor_result_t result;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_aql_block_processor_invoke(
                            &processor, &block.header, &result));
}

TEST(AqlBlockProcessorTest, DirectDispatchPopulatesPacketAndKernarg) {
  DirectDispatchBlock block = MakeDirectDispatchBlock();
  alignas(64) iree_hal_amdgpu_aql_packet_t packets[8] = {};
  iree_hal_amdgpu_aql_ring_t ring = {};
  ring.base = packets;
  ring.mask = 7;
  uint16_t packet_headers[1] = {};
  uint16_t packet_setups[1] = {};
  iree_hal_amdgpu_kernarg_block_t kernarg_blocks[1] = {};
  iree_hal_amdgpu_aql_block_processor_t processor = MakeProcessor(
      &ring, /*packet_count=*/1, packet_headers, packet_setups, kernarg_blocks,
      /*kernarg_block_count=*/1,
      IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_FINAL_PAYLOAD_PACKET);

  iree_hal_amdgpu_aql_block_processor_result_t result;
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_block_processor_invoke(
      &processor, &block.header, &result));

  EXPECT_EQ(result.terminator,
            IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_TERMINATOR_RETURN);
  EXPECT_EQ(result.packets.recorded, 1u);
  EXPECT_EQ(result.packets.emitted, 1u);
  EXPECT_EQ(result.kernargs.consumed, 1u);

  const iree_hal_amdgpu_aql_packet_t& packet = packets[4];
  EXPECT_EQ(packet.dispatch.setup, block.dispatch_command.setup);
  EXPECT_EQ(packet.dispatch.workgroup_size[0],
            block.dispatch_command.workgroup_size[0]);
  EXPECT_EQ(packet.dispatch.grid_size[0], block.dispatch_command.grid_size[0]);
  EXPECT_EQ(packet.dispatch.private_segment_size,
            block.dispatch_command.private_segment_size);
  EXPECT_EQ(packet.dispatch.group_segment_size,
            block.dispatch_command.group_segment_size);
  EXPECT_EQ(packet.dispatch.kernel_object,
            block.dispatch_command.kernel_object);
  EXPECT_EQ(packet.dispatch.kernarg_address, kernarg_blocks[0].data);
  EXPECT_EQ(packet.dispatch.completion_signal.handle, 0u);
  EXPECT_EQ(packet_setups[0], block.dispatch_command.setup);
  EXPECT_EQ(std::memcmp(kernarg_blocks[0].data, block.tail, sizeof(block.tail)),
            0);

  EXPECT_EQ(packet_headers[0],
            iree_hsa_make_packet_header(IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
                                        /*is_barrier=*/true,
                                        IREE_HSA_FENCE_SCOPE_SYSTEM,
                                        IREE_HSA_FENCE_SCOPE_SYSTEM));
}

TEST(AqlBlockProcessorTest,
     PacketHeadersOmitInteriorBarriersWithoutExecutionBarrier) {
  const uint8_t dispatch_command_flags[2] = {
      CommandFlags(
          IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS,
          IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE),
      CommandFlags(
          IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS,
          IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE),
  };
  DispatchBlock<2> block = MakeDispatchBlock(dispatch_command_flags);

  PacketHeaderSummary summary = {};
  IREE_ASSERT_OK(InvokeAndSummarizeDispatchBlock(
      block, IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_FINAL_PAYLOAD_PACKET,
      IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE,
      IREE_HSA_FENCE_SCOPE_NONE, &summary));

  EXPECT_EQ(summary.counts.total, 2u);
  EXPECT_EQ(summary.counts.barrier, 1u);
  EXPECT_FALSE(AqlHeaderHasBarrier(summary.headers.first));
  EXPECT_EQ(AqlHeaderReleaseScope(summary.headers.first),
            IREE_HSA_FENCE_SCOPE_NONE);
  EXPECT_TRUE(AqlHeaderHasBarrier(summary.headers.last));
}

TEST(AqlBlockProcessorTest, PacketHeadersBarrierFirstPayloadForInlineWait) {
  const uint8_t dispatch_command_flags[2] = {
      CommandFlags(
          IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS,
          IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE),
      CommandFlags(
          IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS,
          IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE),
  };
  DispatchBlock<2> block = MakeDispatchBlock(dispatch_command_flags);

  PacketHeaderSummary summary = {};
  IREE_ASSERT_OK(InvokeAndSummarizeDispatchBlock(
      block, IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_FINAL_PAYLOAD_PACKET,
      IREE_HSA_FENCE_SCOPE_AGENT, IREE_HSA_FENCE_SCOPE_NONE,
      IREE_HSA_FENCE_SCOPE_SYSTEM, &summary));

  EXPECT_EQ(summary.counts.total, 2u);
  EXPECT_EQ(summary.counts.barrier, 2u);
  EXPECT_TRUE(AqlHeaderHasBarrier(summary.headers.first));
  EXPECT_EQ(AqlHeaderAcquireScope(summary.headers.first),
            IREE_HSA_FENCE_SCOPE_SYSTEM);
  EXPECT_EQ(AqlHeaderReleaseScope(summary.headers.first),
            IREE_HSA_FENCE_SCOPE_NONE);
  EXPECT_TRUE(AqlHeaderHasBarrier(summary.headers.last));
}

TEST(AqlBlockProcessorTest, PacketHeadersPreserveExplicitMemoryBarrierScopes) {
  const uint8_t dispatch_command_flags[2] = {
      CommandFlags(
          IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS,
          IREE_HSA_FENCE_SCOPE_SYSTEM, IREE_HSA_FENCE_SCOPE_SYSTEM),
      CommandFlags(
          IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS |
              IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_HAS_BARRIER,
          IREE_HSA_FENCE_SCOPE_SYSTEM, IREE_HSA_FENCE_SCOPE_AGENT),
  };
  DispatchBlock<2> block = MakeDispatchBlock(dispatch_command_flags);

  PacketHeaderSummary summary = {};
  IREE_ASSERT_OK(InvokeAndSummarizeDispatchBlock(
      block, IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_FINAL_PAYLOAD_PACKET,
      IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE,
      IREE_HSA_FENCE_SCOPE_NONE, &summary));

  EXPECT_EQ(summary.counts.total, 2u);
  EXPECT_EQ(summary.counts.barrier, 1u);
  EXPECT_FALSE(AqlHeaderHasBarrier(summary.headers.first));
  EXPECT_EQ(AqlHeaderAcquireScope(summary.headers.first),
            IREE_HSA_FENCE_SCOPE_SYSTEM);
  EXPECT_EQ(AqlHeaderReleaseScope(summary.headers.first),
            IREE_HSA_FENCE_SCOPE_SYSTEM);
  EXPECT_TRUE(AqlHeaderHasBarrier(summary.headers.last));
  EXPECT_EQ(AqlHeaderAcquireScope(summary.headers.last),
            IREE_HSA_FENCE_SCOPE_SYSTEM);
  EXPECT_EQ(AqlHeaderReleaseScope(summary.headers.last),
            IREE_HSA_FENCE_SCOPE_AGENT);
}

TEST(AqlBlockProcessorTest, PacketHeadersHonorExplicitExecutionBarrier) {
  const uint8_t dispatch_command_flags[3] = {
      CommandFlags(
          IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS,
          IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE),
      CommandFlags(
          IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS |
              IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_HAS_BARRIER,
          IREE_HSA_FENCE_SCOPE_AGENT, IREE_HSA_FENCE_SCOPE_AGENT),
      CommandFlags(
          IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS,
          IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE),
  };
  DispatchBlock<3> block = MakeDispatchBlock(dispatch_command_flags);

  alignas(64) iree_hal_amdgpu_aql_packet_t packets[8] = {};
  iree_hal_amdgpu_aql_ring_t ring = {};
  ring.base = packets;
  ring.mask = IREE_ARRAYSIZE(packets) - 1u;
  uint16_t packet_headers[3] = {};
  uint16_t packet_setups[3] = {};
  iree_hal_amdgpu_kernarg_block_t kernarg_blocks[3] = {};
  iree_hal_amdgpu_aql_block_processor_t processor = MakeProcessor(
      &ring, /*packet_count=*/3, packet_headers, packet_setups, kernarg_blocks,
      /*kernarg_block_count=*/3,
      IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_FINAL_PAYLOAD_PACKET,
      IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE,
      IREE_HSA_FENCE_SCOPE_NONE);

  iree_hal_amdgpu_aql_block_processor_result_t result;
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_block_processor_invoke(
      &processor, &block.header, &result));
  PacketHeaderSummary summary = SummarizePacketHeaders(packet_headers, 3);

  EXPECT_EQ(summary.counts.total, 3u);
  EXPECT_EQ(summary.counts.barrier, 2u);
  EXPECT_FALSE(AqlHeaderHasBarrier(packet_headers[0]));
  EXPECT_TRUE(AqlHeaderHasBarrier(packet_headers[1]));
  EXPECT_TRUE(AqlHeaderHasBarrier(packet_headers[2]));
}

TEST(AqlBlockProcessorTest,
     PacketHeadersApplySystemAcquireOnlyToFirstDynamicKernargPacket) {
  const uint8_t dispatch_command_flags[2] = {
      CommandFlags(
          IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS,
          IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE),
      CommandFlags(
          IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS,
          IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE),
  };
  DispatchBlock<2> block = MakeDispatchBlock(dispatch_command_flags);

  PacketHeaderSummary summary = {};
  IREE_ASSERT_OK(InvokeAndSummarizeDispatchBlock(
      block, IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_FINAL_PAYLOAD_PACKET,
      IREE_HSA_FENCE_SCOPE_SYSTEM, IREE_HSA_FENCE_SCOPE_NONE,
      IREE_HSA_FENCE_SCOPE_SYSTEM, &summary));

  EXPECT_EQ(summary.counts.total, 2u);
  EXPECT_EQ(summary.counts.barrier, 2u);
  EXPECT_EQ(summary.counts.system_acquire, 1u);
  EXPECT_EQ(summary.counts.system_release, 0u);
}

}  // namespace
}  // namespace iree::hal::amdgpu

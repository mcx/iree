// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_HAL_BASE
#define IREE_DIALECT_HAL_BASE

include "iree/compiler/Dialect/HAL/IR/HALDialect.td"
include "iree/compiler/Dialect/HAL/IR/HALInterfaces.td"
include "iree/compiler/Dialect/Stream/IR/StreamInterfaces.td"
include "iree/compiler/Dialect/Util/IR/UtilTypes.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// HAL types
//===----------------------------------------------------------------------===//

def HAL_Allocator : DialectType<
    HAL_Dialect,
    CPred<"isa<IREE::HAL::AllocatorType>($_self)">,
    "allocator"> {
  let description = [{
    Allocates buffers for a particular device memory space.
  }];
  let builderCall = "$_builder.getType<IREE::HAL::AllocatorType>()";
}

def HAL_Buffer : DialectType<
    HAL_Dialect,
    CPred<"isa<IREE::HAL::BufferType>($_self)">,
    "buffer"> {
  let description = [{
    A memory buffer with a specific memory_type that is used to describe the
    capabilities and behavior of the backing memory of the buffer. Buffers may
    be any mix of host-accessible, host-coherent, or device-accessible for
    various usages. Depending on these memory types the buffers may be mapped
    for access on the host as memory though certain restrictions may be imposed.
  }];
  let builderCall = "$_builder.getType<IREE::HAL::BufferType>()";
}

def HAL_BufferView : DialectType<
    HAL_Dialect,
    CPred<"isa<IREE::HAL::BufferViewType>($_self)">,
    "buffer_view"> {
  let description = [{
    A shaped and typed buffer reference. This just wraps an existing hal.buffer
    with its associated metadata to make it easier to pass across ABI
    boundaries. In most cases buffer views can be elided entirely by the
    compiler and they'll only be seen when calling external functions.
  }];
  let builderCall = "$_builder.getType<IREE::HAL::BufferViewType>()";
}

def HAL_Channel : DialectType<
    HAL_Dialect,
    CPred<"isa<IREE::HAL::ChannelType>($_self)">,
    "collective.channel"> {
  let description = [{
    Channel identifier used to allow for participation in multiple collective
    groups.
  }];
  let builderCall = "$_builder.getType<IREE::HAL::ChannelType>()";
}

def HAL_CommandBuffer : DialectType<
    HAL_Dialect,
    CPred<"isa<IREE::HAL::CommandBufferType>($_self)">,
    "command_buffer"> {
  let description = [{
    Asynchronous command buffer recording interface. Commands are recorded by
    the implementation for later submission to command queues.
  }];
  let builderCall = "$_builder.getType<IREE::HAL::CommandBufferType>()";
}

def HAL_Device : DialectType<
    HAL_Dialect,
    CPred<"isa<IREE::HAL::DeviceType>($_self)">,
    "device"> {
  let description = [{
    Logical device instance.
  }];
  let builderCall = "$_builder.getType<IREE::HAL::DeviceType>()";
}

def HAL_Event : DialectType<
    HAL_Dialect,
    CPred<"isa<IREE::HAL::EventType>($_self)">,
    "event"> {
  let description = [{
    Events are used for defining synchronization scopes within CommandBuffers.
    An event only exists within a single CommandBuffer and must not be used
    across CommandBuffers from the same device or others.
  }];
}

def HAL_Executable : DialectType<
    HAL_Dialect,
    CPred<"isa<IREE::HAL::ExecutableType>($_self)">,
    "executable"> {
  let description = [{
    A prepared and ready-to-dispatch executable.
  }];
  let builderCall = "$_builder.getType<IREE::HAL::ExecutableType>()";
}

def HAL_Fence : DialectType<
    HAL_Dialect,
    CPred<"isa<IREE::HAL::FenceType>($_self)">,
    "fence"> {
  let description = [{
    A set of semaphore timepoints defining a common point in time across
    multiple timelines.
  }];
  let builderCall = "$_builder.getType<IREE::HAL::FenceType>()";
}

def HAL_File : DialectType<
    HAL_Dialect,
    CPred<"isa<IREE::HAL::FileType>($_self)">,
    "buffer"> {
  let description = [{
    A stateless file handle that can be read/written using queue-ordered
    transfer operations.
  }];
  let builderCall = "$_builder.getType<IREE::HAL::FileType>()";
}

def HAL_ObjectType : AnyTypeOf<[
  HAL_Allocator,
  HAL_Buffer,
  HAL_BufferView,
  HAL_CommandBuffer,
  HAL_Device,
  HAL_Event,
  HAL_Executable,
  HAL_Fence,
  HAL_File,
]>;

def HAL_BufferType : AnyTypeOf<[
  HAL_Buffer,
]>;

def HAL_Ordinal : TypeAlias<Index>;
def HAL_OrdinalAttr : Util_IndexAttrBase<"size_t">;
def HAL_OrdinalArrayAttr : TypedArrayAttrBase<HAL_OrdinalAttr, "Array of index ordinal attributes">;

def HAL_ElementType : TypeAlias<I32>;
def HAL_ElementTypeAttr : SignlessIntegerAttrBase<
  I32, "element type attribute">;

def HAL_EncodingType : TypeAlias<I32>;
def HAL_EncodingTypeAttr : SignlessIntegerAttrBase<
  I32, "encoding type attribute">;

def HAL_DeviceSize : TypeAlias<Index>;
def HAL_DeviceSizeAttr : Util_IndexAttrBase<"iree_device_size_t">;
def HAL_DeviceSizes : Variadic<HAL_DeviceSize>;

def HAL_HostSize : TypeAlias<Index>;
def HAL_HostSizeAttr : Util_IndexAttrBase<"size_t">;

def HAL_TimelineValue : TypeAlias<I64>;

def HAL_PrimitiveType : AnyTypeOf<[Index, AnySignlessInteger, AnyFloat, AnyComplex]>;
def HAL_FillPatternType : AnyTypeOf<[I8, I16, I32]>;

def HAL_GlobalRefAttr : Util_AliasedSymbolRefAttr;
def HAL_GlobalType : AnyTypeOf<[HAL_PrimitiveType, AnyVectorOfNonZeroRank, HAL_ObjectType]>;
def HAL_GlobalPtr : Util_PtrOf<HAL_GlobalType>;

def HAL_IndexAttr : Util_IndexAttrBase<"index">;
def HAL_IndexArrayAttr : TypedArrayAttrBase<HAL_IndexAttr,
                                      "index array attribute"> {
  let constBuilderCall = "$_builder.getIndexArrayAttr($0)";
}

def HAL_Dim : TypeAlias<Index>;
def HAL_Dims : Variadic<HAL_Dim>;
def HAL_Shape : Variadic<HAL_Dim>;
def HAL_ShapeDynamicDims : Variadic<HAL_Dim>;

// TODO(benvanik): assert rank 3
def HAL_WorkgroupSizeAttr : TypedArrayAttrBase<
    Util_IndexAttrBase<"size_t">,
    "index array attribute"> {
  let constBuilderCall = "$_builder.getIndexArrayAttr($0)";
}

def HAL_SubgroupSizeAttr : Util_IndexAttrBase<"size_t">;

// A bitmask defining which queues an operation is allowed to execute on.
// The selection is wrapped to the total number of available queues, so 0b0101
// would enable queues 0 and 2 if there were four queues or queue 0 if there
// were two queues.
def HAL_DeviceQueueAffinity : TypeAlias<I64>;

def HAL_DeviceQueuePool : TypeAlias<I64>;

def HAL_DurationMillisAttr : SignlessIntElementsAttr<32> {
  // TODO(b/143184519): add typeDescription support to other things.
  // let description = [{
  //   A duration to wait in milliseconds. 0 indicates that the operation should
  //   return immediately without waiting and can be used as a way to poll handles.
  //   INT32_MAX will wait forever until the handle is signaled.
  // }];
}

def HAL_BufferUsage : TypeAlias<I32>;
def HAL_MemoryType : TypeAlias<I32>;

// Note: This is a copy from the Stream dialect.
// Indicates a value whose lifetime is not yet analyzed.
def HAL_Lifetime_Unknown : I32EnumAttrCase<"Unknown", 0, "*">;
// An externally managed value.
def HAL_Lifetime_External : I32EnumAttrCase<"External", 1, "external">;
// A staging buffer for uploads/downloads.
def HAL_Lifetime_Staging : I32EnumAttrCase<"Staging", 2, "staging">;
// A short-lived value that is used across streams.
def HAL_Lifetime_Transient : I32EnumAttrCase<"Transient", 3, "transient">;
// A long-lived value that persists across streams.
def HAL_Lifetime_Variable : I32EnumAttrCase<"Variable", 4, "variable">;
// An immutable value that persist for the duration of the program.
def HAL_Lifetime_Constant : I32EnumAttrCase<"Constant", 5, "constant">;
def HAL_LifetimeAttr :
    I32EnumAttr<"Lifetime", "IREE HAL value lifetime", [
      HAL_Lifetime_Unknown,
      HAL_Lifetime_External,
      HAL_Lifetime_Staging,
      HAL_Lifetime_Transient,
      HAL_Lifetime_Variable,
      HAL_Lifetime_Constant,
    ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::HAL";
}

//===----------------------------------------------------------------------===//
// Base HAL op classes
//===----------------------------------------------------------------------===//

class HAL_Op<string mnemonic, list<Trait> traits = []> :
    Op<HAL_Dialect, mnemonic, traits> {
  let hasCustomAssemblyFormat = 1;
}

#endif  // IREE_DIALECT_HAL_BASE

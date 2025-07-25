################################################################################
# Autogenerated by build_tools/bazel_to_cmake/bazel_to_cmake.py from           #
# compiler/src/iree/compiler/GlobalOptimization/test/BUILD.bazel               #
#                                                                              #
# Use iree_cmake_extra_content from iree/build_defs.oss.bzl to add arbitrary   #
# CMake-only content.                                                          #
#                                                                              #
# To disable autogeneration for this file entirely, delete this header.        #
################################################################################

iree_add_all_subdirs()

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "cleanup_numeric_narrowing.mlir"
    "conv1x1_to_matmul.mlir"
    "data_layout_propagation.mlir"
    "demote_contraction_inputs_to_bf16.mlir"
    "detach_elementwise_from_named_ops.mlir"
    "expand_tensor_shapes.mlir"
    "fuse_dequantization_matmul.mlir"
    "generalize_named_ops.mlir"
    "global_loop_invariant_code_motion.mlir"
    "hoist_into_globals.mlir"
    "infer_numeric_narrowing.mlir"
    "linalg_quantized_conv_to_conv.mlir"
    "linalg_quantized_matmul_to_matmul.mlir"
    "optimize_numerics.mlir"
    "propagate_linalg_transpose.mlir"
    "raise_special_ops.mlir"
    "remove_zero_extent_tensors.mlir"
    "strided_contraction_to_contraction.mlir"
    "transformation_pipeline.mlir"
    "transpose_and_decompose_concat.mlir"
    "warn_on_uninitialized_values.mlir"
  TOOLS
    FileCheck
    iree-opt
)

### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###

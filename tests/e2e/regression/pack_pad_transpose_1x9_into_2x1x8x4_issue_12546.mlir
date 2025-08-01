func.func @pack_pad_transpose_1x9_into_2x1x8x4_issue_12546.mlir() {
  %iree_input = util.unfoldable_constant
      dense<[[1, 2, 3, 4, 5, 6, 7, 8, 9]]> : tensor<1x9xi8>
  %empty = tensor.empty() : tensor<2x1x8x4xi8>
  %c0_i8 = arith.constant 0 : i8
  %pack = linalg.pack %iree_input padding_value(%c0_i8 : i8) outer_dims_perm = [1, 0]
      inner_dims_pos = [1, 0] inner_tiles = [8, 4] into %empty
      : tensor<1x9xi8> -> tensor<2x1x8x4xi8>
  check.expect_eq_const(%pack, dense<
    [
      [
        [
          [1, 0, 0, 0],
          [2, 0, 0, 0],
          [3, 0, 0, 0],
          [4, 0, 0, 0],
          [5, 0, 0, 0],
          [6, 0, 0, 0],
          [7, 0, 0, 0],
          [8, 0, 0, 0]
        ]
      ],
      [
        [
          [9, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]
        ]
      ]
    ]> : tensor<2x1x8x4xi8>) : tensor<2x1x8x4xi8>
  return
}

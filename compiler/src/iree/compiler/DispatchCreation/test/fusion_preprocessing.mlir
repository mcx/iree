// RUN: iree-opt --iree-dispatch-creation-fusion-preprocessing --split-input-file %s | FileCheck %s

util.func public @fold_insert_slices(%source : tensor<?x?xf32>,
    %dest0 : tensor<?x?xf32>, %dest1 : tensor<?x?xf32>, %val: f32,
    %o1 : index, %o2 : index, %o3 : index, %o4 : index,
    %s1 : index, %s2 : index, %s3 : index, %s4 : index) -> tensor<?x?xf32> {
  %0 = linalg.fill ins(%val : f32) outs(%dest0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = tensor.insert_slice %source into %0[%o1, %o2] [%s1, %s2] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  %2 = linalg.fill ins(%val : f32) outs(%dest1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = tensor.insert_slice %1 into %2[%o3, %o4] [%s3, %s4] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  util.return %3 : tensor<?x?xf32>
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
//      CHECK: func public @fold_insert_slices
// CHECK-SAME:     %[[SOURCE:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[DEST0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[DEST1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[CST:.+]]: f32
// CHECK-SAME:     %[[OFFSET0:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[OFFSET1:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[OFFSET2:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[OFFSET3:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[SIZE0:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[SIZE1:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[SIZE2:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[SIZE3:[a-zA-Z0-9]+]]: index
//      CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[DEST1]] :
//  CHECK-DAG:   %[[NEW_OFFSET0:.+]] = affine.apply #[[MAP]]()[%[[OFFSET0]], %[[OFFSET2]]]
//  CHECK-DAG:   %[[NEW_OFFSET1:.+]] = affine.apply #[[MAP]]()[%[[OFFSET1]], %[[OFFSET3]]]
//      CHECK:   %[[RETURN:.+]] = tensor.insert_slice %[[SOURCE]] into %[[FILL]]
// CHECK-SAME:       [%[[NEW_OFFSET0]], %[[NEW_OFFSET1]]] [%[[SIZE0]], %[[SIZE1]]]
//      CHECK:   util.return %[[RETURN]]

// -----

#ident = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
util.func @single_input_interchange(%arg0: tensor<2x128x128x320xf32>) -> tensor<2x320x128x128xf16> {
  %0 = tensor.empty() : tensor<2x320x128x128xf16>
  %1 = linalg.generic {indexing_maps = [#perm, #ident], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<2x128x128x320xf32>) outs(%0 : tensor<2x320x128x128xf16>) {
  ^bb0(%in: f32, %out: f16):
    %2 = arith.truncf %in : f32 to f16
    linalg.yield %2 : f16
  } -> tensor<2x320x128x128xf16>
  util.return %1 : tensor<2x320x128x128xf16>
}

// CHECK-DAG: #[[$IDENT_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[$PERM_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
// CHECK-LABEL: util.func public @single_input_interchange
// CHECK-SAME:    %[[ARG0:.*]]: tensor<2x128x128x320xf32>
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<2x320x128x128xf16>
// CHECK:         linalg.generic
// CHECK-SAME:      indexing_maps = [#[[$IDENT_MAP]], #[[$PERM_MAP]]]
// CHECK-SAME:      ins(%[[ARG0]] : tensor<2x128x128x320xf32>)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<2x320x128x128xf16>)

// -----

#ident = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
util.func @multi_input_interchange(%arg0: tensor<2x128x128x320xf32>) -> tensor<2x320x128x128xf16> {
  %0 = tensor.empty() : tensor<2x320x128x128xf16>
  %1 = linalg.generic {indexing_maps = [#perm, #perm, #ident], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg0 : tensor<2x128x128x320xf32>, tensor<2x128x128x320xf32>) outs(%0 : tensor<2x320x128x128xf16>) {
  ^bb0(%in: f32, %in_1: f32, %out: f16):
    %2 = arith.addf %in, %in_1 : f32
    %3 = arith.truncf %2 : f32 to f16
    linalg.yield %3 : f16
  } -> tensor<2x320x128x128xf16>
  util.return %1 : tensor<2x320x128x128xf16>
}

// CHECK-DAG: #[[$IDENT_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[$PERM_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
// CHECK-LABEL: util.func public @multi_input_interchange
// CHECK-SAME:    %[[ARG0:.*]]: tensor<2x128x128x320xf32>
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<2x320x128x128xf16>
// CHECK:         linalg.generic
// CHECK-SAME:      indexing_maps = [#[[$IDENT_MAP]], #[[$IDENT_MAP]], #[[$PERM_MAP]]]
// CHECK-SAME:      ins(%[[ARG0]], %[[ARG0]] : tensor<2x128x128x320xf32>, tensor<2x128x128x320xf32>)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<2x320x128x128xf16>)

// -----

#ident = affine_map<(d0, d1) -> (d0, d1)>
#perm0 = affine_map<(d0, d1) -> (d1, d0)>
util.func @multi_input_no_interchange(%arg0: tensor<10x10xf32>) -> tensor<10x10xf16> {
  %0 = tensor.empty() : tensor<10x10xf16>
  %1 = linalg.generic {indexing_maps = [#ident, #perm0, #perm0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg0 : tensor<10x10xf32>, tensor<10x10xf32>) outs(%0 : tensor<10x10xf16>) {
  ^bb0(%in: f32, %in_1: f32, %out: f16):
    %2 = arith.addf %in, %in_1 : f32
    %3 = arith.truncf %2 : f32 to f16
    linalg.yield %3 : f16
  } -> tensor<10x10xf16>
  util.return %1 : tensor<10x10xf16>
}

// CHECK-DAG: #[[$IDENT_MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[$PERM_MAP0:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-LABEL: util.func public @multi_input_no_interchange
// CHECK-SAME:    %[[ARG0:.*]]: tensor<10x10xf32>
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<10x10xf16>
// CHECK:         linalg.generic
// CHECK-SAME:      indexing_maps = [#[[$IDENT_MAP]], #[[$PERM_MAP0]], #[[$PERM_MAP0]]]
// CHECK-SAME:      ins(%[[ARG0]], %[[ARG0]] : tensor<10x10xf32>, tensor<10x10xf32>)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<10x10xf16>)

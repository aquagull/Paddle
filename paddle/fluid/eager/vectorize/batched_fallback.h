// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <numeric>
#include <tuple>
#include <type_traits>
#include "paddle/phi/api/include/tensor.h"
#include "paddle/utils/small_vector.h"

namespace paddle {
namespace vmap {
static paddle::small_vector<indexing::TensorIndex, kVmapStaticDimVecSize>
computeIndex(int64_t linear_idx, IntArrayRef sizes) {
  paddle::small_vector<indexing::TensorIndex, kVmapStaticDimVecSize> result;
  result.reserve(sizes.size());
  for (auto it = sizes.rbegin(); it != sizes.rend(); it++) {
    auto remainder = linear_idx % *it;
    result.push_back(remainder);
    linear_idx -= remainder;
    linear_idx /= *it;
  }
  std::reverse(std::begin(result), std::end(result));
  return result;
}

template <typename>
struct is_tuple : std::false_type {};

template <typename... T>
struct is_tuple<std::tuple<T...>> : std::true_type {};

template <typename T>
struct all_tensors : std::false_type {};

template <>
struct all_tensors<paddle::Tensor> : std::true_type {};

template <typename... Ts>
struct all_tensors<std::tuple<Ts...>> : std::conjunction<all_tensors<Ts>...> {};
};  // namespace vmap
};  // namespace paddle

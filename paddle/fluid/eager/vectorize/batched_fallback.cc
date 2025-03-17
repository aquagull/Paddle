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

#include "paddle/fluid/eager/vectorize/batched_fallback.h"
#include <sstream>
#include <typeinfo>
#include <vector>
#include "paddle/fluid/eager/vectorize/vmap_transforms.h"
#include "paddle/phi/core/batched_tensor.h"
namespace paddle {
namespace vmap {
template <typename Func, typename... Args>
auto batchedTensorForLoopFallback(Func kernel, Args &&...args) {
  using ResultType = decltype(kernel(std::declval<Args>()...));

  std::vector<int64_t> batch_sizes;
  std::vector<int64_t> bdims;
  (
      [&](auto &&arg) {
        if constexpr (is_batched_v<std::decay_t<decltype(arg)>>) {
          if (auto *batched = maybeGetBatchedImpl(arg)) {
            if (bdims.empty()) {
              bdims = batched->bdims();
              const auto &physical_shape = batched->value().sizes();
              for (auto dim : bdims) {
                batch_sizes.emplace_back(physical_shape[dim]);
              }
            }
          }
        }
      }(std::forward<Args>(args)),
      ...);
  // 计算总批次数
  const int64_t num_batches = std::accumulate(
      batch_sizes.begin(), batch_sizes.end(), 1LL, std::multiplies<int64_t>());
  VmapPhysicalView input_physical_batch = nullptr;
  // 准备分片参数
  auto prepare_sliced_args = [&](int64_t linear_idx) {
    return std::make_tuple([&](auto &&arg) {
      int64_t index = computeIndex(linear_idx, batch_sizes);
      if constexpr (is_batched_v<std::decay_t<decltype(arg)>>) {
        auto *batched = maybeGetBatchedTensorImpl(arg);
        if (input_physical_batch == nullptr) {
          input_physical_batch = logicalToPhysical(batched);
        }
        return logicalToPhysical(batched).tensor().index(index);
      } else {
        return arg;
      }
    }(std::forward<Args>(args))...);
  };

  // 逐批次处理
  std::vector<ResultType> output_shards(num_batches);
  for (int64_t i = 0; i < num_batches; i++) {
    auto sliced_args = prepare_sliced_args(i);
    output_shards[i] = std::move(std::apply(kernel, sliced_args));
  }

  // 合并结果
  // output_shards 的结构会形如 [[out1_b0, out2_b0...], [out1_b1, out2_b1...],
  // ..., [out1_bN, out2_bN...]] 需要将 ​每个返回值（out1,out2,
  // ...）的对应分片独立合并，最终返回 [merged_out1, merged_out2...]
  if constexpr (std::is_same_v<ResultType, Tensor>) {
    auto stacked = stack(output_shards);  // 堆叠结果
    VmapDimVector output_sizes(batch_sizes);
    output_sizes.insert(
        output_sizes.end(), stacked.sizes().begin() + 1, stacked.sizes().end());
    return input_physical_batch.getPhysicalToLogicalMap().apply(
        stacked.view(output_sizes));
  } else if (std::is_same_v<std::decay_t<ResultType>, std::tuple<Args...>>) {
    size_t num_returns = std::tuple_size_v<ResultType>;
    std::array<std::vector<Tensor>, num_returns> output_groups;

    for (const auto &shard : output_shards) {
      std::apply(
          [&](const auto &...ts) {
            size_t i = 0;
            ((output_groups[i++].push_back(ts)), ...);
          },
          shard);
    }

    return std::apply(
        [&](auto &...groups) {
          return std::make_tuple([&] {
            auto stacked = stack(groups);
            VmapDimVector output_sizes(batch_sizes);
            output_sizes.insert(output_sizes.end(),
                                stacked.sizes().begin() + 1,
                                stacked.sizes().end());
            return input_physical_batch.getPhysicalToLogicalMap().apply(
                stacked.view(output_sizes));
          }()...);
        },
        output_groups);
  } else {
    std::stringstream ss;
    ss << "Invalid ResultType: " << typeid(ResultType).name();
    PD_THROW(ss.str());
  }
}

};  // namespace vmap
};  // namespace paddle

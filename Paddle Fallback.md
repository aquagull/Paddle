# Paddle Fallback

## 背景

当张量类型为BatchTensorImpl时，并且算子对应的Batch_rules未被实现的时候，将会执行Fallback。

## torch

为所有the dispatchkey is BatchedTensorImpl都注册fallback函数**batchedTensorForLoopFallback**

```cpp
TORCH_LIBRARY_IMPL(_, Batched, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&batchedTensorForLoopFallback>());
}
```

```cpp
void batchedTensorForLoopFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const auto num_returns = schema.returns().size(); // 这里获取返回值个数，用于创建output
  const auto num_arguments = schema.arguments().size(); // 获得参数的个数 
  const auto arguments = torch::jit::last(stack, num_arguments); // 将指针指向参数
  
  // 如果返回值有非张量或者张量列表，就抛出错误
  TORCH_CHECK(areAllReturnsTensors(schema) && !areAnyArgumentsTensorList(schema),
              "Batching rule not implemented for ", schema.operator_name(), ". ",
              "We could not generate a fallback.");
	
	// 若无参数参与批处理，直接调用原始操作符
  if (std::none_of(arguments.begin(), arguments.end(), ivalueParticipatesInCurrentLevel)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    op.callBoxed(stack);
    return;
  }
  
	// 调用inplace的fallback
  if (isInplaceOp(schema)) {
    batchedTensorInplaceForLoopFallback(op, stack);
    return;
  }
  TORCH_CHECK(!schema.is_mutable() && !schema.hasAnyAliasInfo(),
              "Batching rule not implemented for ", schema.operator_name(), "; ",
              "the fallback path doesn't work on out= or view ops.");
  // 至少有一个返回值
  TORCH_CHECK(num_returns >= 1,
              "Batching rule not implemented for ", schema.operator_name(), ". ",
              "The fallback path does not support operations with no returns.");
  warnFallback(schema, /*in_place*/false);

  const auto arguments_begin = stack->size() - num_arguments;

  // 遍历参数，通过maybeGetBatchedImpl找到BatchTensor/
  // 记录在batched_tensor_inputs、batched_tensor_inputs_position
  at::SmallVector<Tensor,kVmapTransformStaticInputSize> batched_tensor_inputs;
  VmapDimVector batched_tensor_inputs_position;
  for (const auto idx : c10::irange(0, arguments.size())) {
    const auto& ivalue = arguments[idx];
    if (!ivalue.isTensor()) {
      continue;
    }
    const auto& tensor = ivalue.toTensor();
    if (!tensor.defined()) {
      continue;
    }
    const auto* batched = maybeGetBatchedImpl(tensor);
    if (!batched) {
      continue;
    }
    batched_tensor_inputs.push_back(tensor);
    batched_tensor_inputs_position.push_back(static_cast<int64_t>(idx));
  }
  TORCH_INTERNAL_ASSERT(!batched_tensor_inputs.empty());

  // 将Logical view 转成 Physical view
  const auto input_physical_views = MultiBatchVmapTransform::logicalToPhysical(
      batched_tensor_inputs);

  // 计算 总批次数
  // 1. 有多少个批次维度
  auto num_batch_dims = input_physical_views.front().numBatchDims();
  // 2. 张量的维度大小数组 [B1, B2, a, b, c]
  auto some_sizes = input_physical_views.front().tensor().sizes();
  // 3. 获得[B1, B2]
  auto batch_sizes = ArrayRef<int64_t>(some_sizes.begin(), some_sizes.begin() + num_batch_dims);
  // 4. num_batches = B1 * B2
  const auto num_batches = c10::multiply_integers(batch_sizes);
  // Without a shape-checking API, we're unable to compute the correct shape of
  // the output so we just error out.
  TORCH_CHECK(num_batches > 0,
      "Batching rule not implemented for ", schema.operator_name(), ". ",
      "The fallback path does not support vmap over dims of size 0.");

  // Strategy: For each batch, we are going to push slices (where applicable)
  // of the arguments onto `stack`, call `op`, and store the result in
  // `output_shards`.
  //
  // NOTE: [Output shards layout]
  // Assume that the operator has three outputs: a, b, c.
  // The layout of output_shards is as follows:
  // [ a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3]
  // This is so that we can call at::stack([a0...a3]), at::stack([b0...b3])
  // more easily in the next step.
  std::vector<Tensor> output_shards(num_batches * num_returns);
	// 逐批次计算，并收集结果到output_shards
  for (int64_t linear_idx = 0; linear_idx < num_batches; ++linear_idx) {
    auto index = computeIndex(linear_idx, batch_sizes);
    auto batched_tensor_inputs_pos_iter = batched_tensor_inputs_position.begin();
    auto input_physical_views_iter = input_physical_views.begin();
    for (const auto arg_idx : c10::irange(0, num_arguments)) {
      // We assume that torch::jit::Stack is backed by vector<IValue> for
      // simplicity. When that is not the case, this code should be updated.
      const auto& argument = (*stack)[arguments_begin + arg_idx];
      if (batched_tensor_inputs_pos_iter == batched_tensor_inputs_position.end()
          || (int64_t)arg_idx != *batched_tensor_inputs_pos_iter) {
        // argument isn't a BatchedTensor
        torch::jit::push(stack, argument);
        continue;
      }
      // argument is a BatchedTensor
      TORCH_INTERNAL_ASSERT(input_physical_views_iter != input_physical_views.end());
      const auto& physical_view_for_argument = *input_physical_views_iter;
      c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
      torch::jit::push(stack, physical_view_for_argument.tensor().index(index));
      batched_tensor_inputs_pos_iter++;
      input_physical_views_iter++;
    }

    // std::cout << "[Fallback]: ";
    // at::dump_tensor((*stack)[stack->size() - 1].toTensor());
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    op.callBoxed(stack);

    // Store the result into `output_shards`. See NOTE: [Output shards layout]
    // to learn about the details of how we store the shards.
    const auto returns = torch::jit::last(stack, num_returns);
    for (const auto  return_idx : c10::irange(0, returns.size())) {
      output_shards[num_batches * return_idx + linear_idx] = returns[return_idx].toTensor();
    }
    torch::jit::drop(stack, num_returns);
  }

  // For each output Tensor, stack the shards of the tensor together to form a return
  torch::jit::drop(stack, num_arguments);
  // 堆叠结果
  auto output_shards_chunks = MatrixRef<Tensor>(output_shards, num_batches);
  for (const auto return_idx : c10::irange(0, num_returns)) {
    auto shards = output_shards_chunks[return_idx];
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    auto flat_output = safeStack(shards);
    // See NOTE [vmap through backward and undefined grad]
    if (!flat_output.defined()) {
      torch::jit::push(stack, flat_output);
      continue;
    }
    VmapDimVector output_sizes(batch_sizes);
    output_sizes.insert(
        output_sizes.end(),
        flat_output.sizes().begin() + 1,
        flat_output.sizes().end());
    torch::jit::push(
        stack,
        input_physical_views.front().getPhysicalToLogicalMap().apply(flat_output.view(output_sizes)));
  }
}
```

## 预期

希望能够通过以下代码，调用fallback

`auto result = fallback(paddle::experimental::op, x, y, ...);`

Fallback流程应为：

1. 各种检查，其中包括
    - 返回值是否全部都是Tensor。
    - 是否包含TensorList。
    - 不支持 `out=` 参数或视图操作（如 `schema.is_mutable()` 检查）。
    - 返回值数量必须 ≥ 1（`num_returns >= 1`）
2. 获取所有的BatchTensor的bdims，从而计算出num_batch。以及将BatchTensor从逻辑视图转为物理视图。
3. "拆开"Args，将其中的BatchTensor切片，并构造新的Args，具体为：
    - 对每个参数，如果是 `BatchedTensor`，取其当前批次的物理视图切片（`tensor.index(index)`）。
    - 非 `BatchedTensor` 直接传递原值。
    
    Args作为参数直接传递给算子，并存储每个分片的输出
    
4. 将输出堆叠，恢复逻辑视图，返回
    1. output_shards 的结构会形如 `[[out1_b0, out2_b0...], [out1_b1, out2_b1...], ..., [out1_bN, out2_bN...]]`，需要将 每个返回值（out1,out2, ...）的对应分片独立合并，最终返回 `[merged_out1, merged_out2...]`

## 代码实现

```cpp
template <typename>
struct is_tuple : std::false_type
{
};

template <typename... T>
struct is_tuple<std::tuple<T...>> : std::true_type
{
};

template <typename T>
struct all_tensors : std::false_type
{
};

template <>
struct all_tensors<Tensor> : std::true_type
{
};

template <typename... Ts>
struct all_tensors<std::tuple<Ts...>> : std::conjunction<all_tensors<Ts>...>
{
};

template <typename T>
constexpr bool num_returns_valid()
{
    if constexpr (std::is_same_v<T, Tensor>)
        return true;
    else if constexpr (is_tuple<T>::value)
        return std::tuple_size_v<T> >= 1 && all_tensors<T>::value; // 元组非空且全为Tensor
    return false;
};

template <typename T>
constexpr bool is_tensorlist = std::is_same_v<T, std::vector<Tensor>>;

template <typename... Args>
constexpr bool contains_tensorlist_v = (is_tensorlist<Args> || ...);

template <typename Func, typename... Args>
auto Fallback(Func kernel, Args &&...args)
{
    using ResultType = decltype(kernel(std::declval<Args>()...));
    // PD_CHECK(num_returns_valid<ResultType>(), "");
    // PD_CHECK(!contains_tensorlist_v<Args...>, "");
    num_returns_valid<ResultType>();
    contains_tensorlist_v<Args...>;
    std::vector<int64_t> batch_sizes;
    std::vector<int64_t> bdims;
    ([&](auto &&arg)
     {
         if constexpr (is_batched_v<std::decay_t<decltype(arg)>>)
         {
             if (auto *batched = maybeGetBatchedImpl(arg))
             {
                 if (bdims.empty())
                 {
                     bdims = batched->bdims();
                     const auto &physical_shape = batched->value().sizes();
                     for (auto dim : bdims)
                     {
                         batch_sizes.emplace_back(physical_shape[dim]);
                     }
                 }
             }
         } }(std::forward<Args>(args)), ...);

    // if(InplaceOp()){
    // batchedTensorInplaceForLoopFallback();
    // }

    // 计算总批次数
    const int64_t num_batches = std::accumulate(batch_sizes.begin(), batch_sizes.end(), 1LL, std::multiplies<int64_t>());
    VmapPhysicalView input_physical_batch = nullptr;
    // 准备分片参数
    auto prepare_sliced_args = [&](int64_t linear_idx)
    {
        return std::make_tuple([&](auto &&arg)
                               {
            int64_t index = computeIndex(linear_idx, batch_sizes);
            if constexpr (is_batched_v<std::decay_t<decltype(arg)>>)
            {
                auto *batched = maybeGetBatchedTensorImpl(arg);
                if (input_physical_batch == nullptr)
                {
                    input_physical_batch = logicalToPhysical(batched);
                }
                return logicalToPhysical(batched).tensor().index(index);
            }
            else
                return arg; }(std::forward<Args>(args))...);
    };

    // 逐批次处理
    std::vector<ResultType> output_shards(num_batches);
    for (int64_t i = 0; i < num_batches; i++)
    {
        auto sliced_args = prepare_sliced_args(i);
        output_shards[i] = std::move(std::apply(kernel, sliced_args));
    }

    // 合并结果
    // output_shards 的结构会形如 [[out1_b0, out2_b0...], [out1_b1, out2_b1...], ..., [out1_bN, out2_bN...]]
    // 需要将 每个返回值（out1,out2, ...）的对应分片独立合并，最终返回 [merged_out1, merged_out2...]
    if constexpr (std::is_same_v<ResultType, Tensor>)
    {
        auto stacked = stack(output_shards); // 堆叠结果
        VmapDimVector output_sizes(batch_sizes);
        output_sizes.insert(
            output_sizes.end(),
            stacked.sizes().begin() + 1,
            stacked.sizes().end());
        return input_physical_batch.getPhysicalToLogicalMap().apply(stacked.view(output_sizes));
    }
    else if constexpr (std::is_same_v<std::decay_t<ResultType>, std::tuple<Args...>>)
    {
        constexpr size_t num_returns = std::tuple_size_v<ResultType>;
        std::array<std::vector<Tensor>, num_returns> output_groups;

        for (const auto &shard : output_shards)
        {
            std::apply([&](const auto &...ts)
                       {
                size_t i = 0;
                ((output_groups[i++].push_back(ts)), ...); }, shard);
        }

        return std::apply([&](auto &...groups)
                          { return std::make_tuple([&]
                                                   {
                auto stacked = stack(groups);
                VmapDimVector output_sizes(batch_sizes);
                output_sizes.insert(
                    output_sizes.end(),
                    stacked.sizes().begin() + 1,
                    stacked.sizes().end()
                ) ;
                return input_physical_batch.getPhysicalToLogicalMap().apply(
                    stacked.view(output_sizes)
                ); }()...); }, output_groups);
    }
}
```
## Reference

https://github.com/PaddlePaddle/community/blob/master/pfcc/paddle-code-reading/kernel_selection/20221130_kernel_selection.md
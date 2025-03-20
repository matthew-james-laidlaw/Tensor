#pragma once

#include <array>

template <typename T1, typename T2, size_t Order, size_t Dim = 0>
auto CopyElementwiseRecursive(const T1& src, T2& dst, std::array<size_t, Order>& indices) -> void
{
    if constexpr (Dim == Order)
    {
        dst(indices) = static_cast<T2>(src(indices));
    }
    else
    {
        for (size_t i = 0; i < src.shape[Dim]; ++i)
        {
            indices[Dim] = i;
            CopyElementwiseRecursive<T1, T2, Order, Dim + 1>(src, dst, indices);
        }
    }
}

template <typename T1, typename T2, size_t Order>
auto CopyElementwise(const T1& src, T2& dst) -> void
{
    Expect(src.Shape() == dst.Shape());
    auto indices = std::array<size_t, Order>{0};
    CopyElementwiseRecursive(src, dst, indices);
}

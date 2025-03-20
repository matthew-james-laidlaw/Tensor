#pragma once

#include "Expect.hpp"
#include "TensorLike.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <numeric>

template <size_t Order>
auto GetSize(std::array<size_t, Order> const& shape) -> size_t
{
    return std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<>());
}

template <size_t Order>
auto GetStrides(std::array<size_t, Order> const& shape) -> std::array<size_t, Order>
{
    std::array<size_t, Order> strides{};
    if constexpr (Order > 0)
    {
        strides[Order - 1] = 1;
        for (size_t i = Order - 1; i-- > 0;)
        {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    return strides;
}

// provides all the information needed to iterate through a
// (potentially non-contiguous) view over a block of memory
template <typename T, size_t Order>
struct StridedArrayView
{

    T* data;
    std::array<size_t, Order> shape;
    std::array<size_t, Order> strides;

    inline auto operator()(std::array<size_t, Order> const& indices) -> T&
    {
        return data[GetLinearIndex(indices)];
    }

    inline auto operator()(std::array<size_t, Order> const& indices) const -> T
    {
        return data[GetLinearIndex(indices)];
    }

private:

    inline auto GetLinearIndex(std::array<size_t, Order> const& indices) const -> size_t
    {
        return std::inner_product(indices.begin(), indices.end(), strides.begin(), static_cast<size_t>(0));
    }
};

template <typename T1, typename T2, size_t Order, size_t Dim = 0>
auto CopyElementwiseRecursive(const StridedArrayView<T1, Order>& src,
                              StridedArrayView<T2, Order>& dst,
                              std::array<size_t, Order>& indices) -> void
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

/** @brief Performs element-wise copy by iterating over two arrays of data
 * which may be non-contiguous or of different but convertible types, thus they must be iterated over
 * with the given strides to locate elements and perform the copy
 * rather than copying contiguous chunks of memory.
 */
template <typename T1, typename T2, size_t Order>
    requires TensorLike<T1, Order> &&
             TensorLike<T2, Order> &&
             std::convertible_to<typename T1::ValueType, typename T2::ValueType>
auto CopyElementwise(const T1& src, T2& dst) -> void
{
    Expect(src.Shape() == dst.Shape());

    auto srcView = StridedArrayView<const typename T1::ValueType, Order>{src.Data(), src.Shape(), src.Strides()};
    auto dstView = StridedArrayView<typename T2::ValueType, Order>{dst.Data(), dst.Shape(), dst.Strides()};

    auto indices = std::array<size_t, Order>{0};

    CopyElementwiseRecursive(srcView, dstView, indices);
}

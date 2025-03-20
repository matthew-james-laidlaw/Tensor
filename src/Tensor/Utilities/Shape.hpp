#pragma once

#include <array>

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

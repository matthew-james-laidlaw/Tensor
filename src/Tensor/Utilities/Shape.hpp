#pragma once

#include <array>

template <size_t Order>
auto GetSize(std::array<size_t, Order> const& shape) -> size_t
{
    return std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<>());
}

template <size_t Order>
auto GetStrides(const std::array<size_t, Order>& shape) -> std::array<size_t, Order>
{
    std::array<size_t, Order> strides{};
    std::exclusive_scan(shape.rbegin(), shape.rend(), strides.rbegin(), static_cast<size_t>(1), std::multiplies<>());
    return strides;
}

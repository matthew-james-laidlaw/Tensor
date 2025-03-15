#pragma once

#include <algorithm>
#include <array>
#include <initializer_list>
#include <numeric>

template <size_t Order>
auto ComputeStrides(std::array<size_t, Order> const& shape) -> std::array<size_t, Order>
{
    auto strides = std::array<size_t, Order>();
    std::transform(shape.rbegin(), shape.rend(), strides.rbegin(),
                   [&, product = size_t(1)](size_t dimension) mutable -> size_t
    {
        size_t current = product;
        product *= dimension;
        return current;
    });
    return strides;
}

template <size_t Order>
auto ComputeSize(std::array<size_t, Order> const& shape) -> size_t
{
    return std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<>());
}

template <size_t Order, typename... Ts>
concept ValidShapeOrIndex = (sizeof...(Ts) == Order) && (std::convertible_to<Ts, size_t> && ...);

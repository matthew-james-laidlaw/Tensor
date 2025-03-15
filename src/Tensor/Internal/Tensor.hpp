#pragma once

#include "ITensor.hpp"

#include <numeric>
#include <vector>

template <size_t Order>
auto ComputeStrides(std::array<size_t, Order> const& shape) -> std::array<size_t, Order>
{
    std::array<size_t, Order> strides;
    std::exclusive_scan(shape.rbegin(), shape.rend(), strides.rbegin(), size_t(1), std::multiplies<>());
    return strides;
}

template <size_t Order>
auto ComputeSize(std::array<size_t, Order> const& shape) -> size_t
{
    return std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
}

template <typename T, size_t Order>
class Tensor : public ITensor<Tensor<T, Order>>
{
private:

    std::array<size_t, Order> shape_;
    std::array<size_t, Order> strides_;
    std::vector<T> data_;

public:

    using ValueType = T;
    static constexpr size_t kOrder = Order;

    Tensor(std::array<size_t, Order> const& shape)
        : shape_(shape)
        , strides_(ComputeStrides(shape))
        , data_(ComputeSize(shape))
    {}

    auto Shape() const -> std::array<size_t, Order>
    {
        return shape_;
    }

    auto Strides() const -> std::array<size_t, Order>
    {
        return strides_;
    }

    auto Data() -> T*
    {
        return data_.data();
    }

    inline auto operator()(std::array<size_t, Order> const& indices) -> T&
    {
        return data_[LinearIndex(indices)];
    }

    inline auto operator()(std::array<size_t, Order> const& indices) const -> T
    {
        return data_[LinearIndex(indices)];
    }

private:

    inline auto LinearIndex(std::array<size_t, Order> const& indices) const -> size_t
    {
        return std::inner_product(indices.begin(), indices.end(), strides_.begin(), size_t(0));
    }
};

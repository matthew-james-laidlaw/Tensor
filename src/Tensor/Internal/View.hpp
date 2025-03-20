#pragma once

#include "TensorLike.hpp"

template <typename T, size_t Order>
class View
{
private:

    std::array<size_t, Order> mShape;
    std::array<size_t, Order> mStrides;
    size_t mOffset;
    T* mData;

public:

    using ValueType = T;
    static constexpr size_t kOrder = Order;

    View(T* data, std::array<size_t, Order> const& shape, std::array<size_t, Order> const& strides, size_t offset)
        : mShape(shape)
        , mStrides(strides)
        , mOffset(offset)
        , mData(data)
    {}

    auto Data() -> T*
    {
        return mData;
    }

    auto Data() const -> const T*
    {
        return mData;
    }

    auto Shape() const -> std::array<size_t, Order>
    {
        return mShape;
    }

    auto Strides() const -> std::array<size_t, Order>
    {
        return mStrides;
    }

    auto operator()(std::array<size_t, Order> const& indices) -> T&
    {
        return mData[GetLinearIndex(indices)];
    }

    auto operator()(std::array<size_t, Order> const& indices) const -> T
    {
        return mData[GetLinearIndex(indices)];
    }

private:

    inline auto GetLinearIndex(std::array<size_t, Order> const& indices) const -> size_t
    {
        return std::inner_product(indices.begin(), indices.end(), mStrides.begin(), static_cast<size_t>(0));
    }
};

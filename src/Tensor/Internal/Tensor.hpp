#pragma once

#include "TensorLike.hpp"

#include <memory>
#include <numeric>

template <typename T, size_t Order>
class Tensor
{
private:

    std::array<size_t, Order> mShape;
    std::array<size_t, Order> mStrides;
    size_t mSize;
    std::unique_ptr<T[]> mData;

public:

    using ValueType = T;
    static constexpr size_t kOrder = Order;

    Tensor(std::array<size_t, Order> const& shape)
        : mShape(shape)
        , mSize(GetSize(mShape))
        , mStrides(GetStrides(mShape))
        , mData(new T[mSize])
    {}

    Tensor(std::array<size_t, Order> const& shape, T fill)
        : Tensor(shape)
    {
        std::fill_n(mData.get(), mSize, fill);
    }

    Tensor(Tensor const& other)
        : Tensor(other.Shape())
    {
        std::copy_n(other.mData.get(), mSize, mData.get());
    }

    Tensor(Tensor&& other) noexcept
        : mShape(other.mShape)
        , mSize(other.mSize)
        , mStrides(other.mStrides)
        , mData(std::move(other.mData))
    {}

    Tensor& operator=(Tensor const& other)
    {
        if (this != &other)
        {
            mShape = other.mShape;
            mSize = other.mSize;
            mStrides = other.mStrides;
            mData.reset(new T[mSize]);
            std::copy_n(other.mData.get(), mSize, mData.get());
        }
        return *this;
    }

    Tensor& operator=(Tensor&& other) noexcept
    {
        if (this != &other)
        {
            mShape = other.mShape;
            mSize = other.mSize;
            mStrides = other.mStrides;
            mData = std::move(other.mData);
        }
        return *this;
    }

    template <typename Other>
        requires TensorLike<Other, Order> && std::convertible_to<typename Other::ValueType, ValueType>
    Tensor(Other const& other)
        : Tensor(other.Shape())
    {
        CopyElementwise<Other, std::remove_reference_t<decltype(*this)>, Order>(other, *this);
    }

    auto Data() -> T*
    {
        return mData.get();
    }

    auto Data() const -> const T*
    {
        return mData.get();
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

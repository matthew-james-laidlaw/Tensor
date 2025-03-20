#pragma once

#include "../Mixins/Indexable.hpp"
#include "../Mixins/Printable.hpp"
#include "../Mixins/Sliceable.hpp"
#include "../Utilities/Copy.hpp"
#include "../Utilities/Shape.hpp"

#include <algorithm>
#include <memory>

template <typename T, size_t Order>
class Tensor : public Indexable<Tensor<T, Order>>,
               public Printable<Tensor<T, Order>>,
               public Sliceable<Tensor<T, Order>, Order>
{
private:

    std::array<size_t, Order> mShape;
    std::array<size_t, Order> mStrides;
    size_t mSize;
    std::unique_ptr<T[]> mData;

public:

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
};

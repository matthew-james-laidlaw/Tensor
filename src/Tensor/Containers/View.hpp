#pragma once

#include "../Mixins/Indexable.hpp"
#include "../Mixins/Printable.hpp"
#include "../Mixins/Sliceable.hpp"

template <typename T, size_t Order>
class View : public OffsetIndexable<View<T, Order>, T, Order>,
             public Printable<View<T, Order>, Order>,
             public Sliceable<View<T, Order>, Order>
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

    inline auto Offset() const -> size_t
    {
        return mOffset;
    }
};

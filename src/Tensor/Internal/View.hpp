#pragma once

template <typename T, size_t Order>
class View : public ITensor<View<T, Order>>
{
private:

    T* data_;
    std::array<size_t, Order> shape_;
    std::array<size_t, Order> strides_;
    size_t offset_;

public:

    using ValueType = T;
    static constexpr size_t kOrder = Order;

    View(T* data, std::array<size_t, Order> const& shape, std::array<size_t, Order> strides, size_t offset)
        : data_(data)
        , shape_(shape)
        , strides_(strides)
        , offset_(offset)
    {}

    auto Data() -> T*
    {
        return data_;
    }

    auto Data() const -> const T*
    {
        return data_;
    }

    auto Shape() const -> std::array<size_t, Order>
    {
        return shape_;
    }

    auto Strides() const -> std::array<size_t, Order>
    {
        return strides_;
    }

    auto Offset() const -> size_t
    {
        return offset_;
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
        return std::inner_product(indices.begin(), indices.end(), strides_.begin(), size_t(0)) + offset_;
    }
};

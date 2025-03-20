#pragma once

#include <array>
#include <numeric>

template <typename Derived, size_t Order>
class Indexable
{
protected:

    inline auto ComputeLinearIndex(std::array<size_t, Order> const& indices) const -> size_t
    {
        auto& self = static_cast<Derived const&>(*this);
        return std::inner_product(indices.begin(), indices.end(), self.Strides().begin(), static_cast<size_t>(0));
    }
};

template <typename Derived, typename T, size_t Order>
class DirectIndexable : public Indexable<Derived, Order>
{
public:

    inline auto operator()(std::array<size_t, Order> const& indices) -> T&
    {
        auto& self = static_cast<Derived&>(*this);
        return self.Data()[this->ComputeLinearIndex(indices)];
    }

    inline auto operator()(std::array<size_t, Order> const& indices) const -> T const&
    {
        auto& self = static_cast<Derived const&>(*this);
        return self.Data()[this->ComputeLinearIndex(indices)];
    }
};

template <typename Derived, typename T, size_t Order>
class OffsetIndexable : public Indexable<Derived, Order>
{
public:

    inline auto operator()(std::array<size_t, Order> const& indices) -> T&
    {
        auto& self = static_cast<Derived&>(*this);
        return self.Data()[this->ComputeLinearIndex(indices) + self.Offset()];
    }

    inline auto operator()(std::array<size_t, Order> const& indices) const -> T const&
    {
        auto& self = static_cast<Derived const&>(*this);
        return self.Data()[this->ComputeLinearIndex(indices) + self.Offset()];
    }
};

#pragma once

#include "../Containers/Traits.hpp"

#include <array>
#include <cstddef>
#include <numeric>

/*
 * Indexable mixin for tensor-like classes.
 * Provides multidimensional indexing operator().
 */
template <typename Derived>
class Indexable
{
public:

    using value_type = TensorTraits<Derived>::value_type;
    static constexpr size_t order = TensorTraits<Derived>::order;
    static constexpr bool has_offset = TensorTraits<Derived>::has_offset;

    inline auto operator()(std::array<size_t, order> const& indices) -> value_type&
    {
        auto& self = static_cast<Derived&>(*this);
        return self.Data()[this->ComputeLinearIndex(indices)];
    }

    inline auto operator()(std::array<size_t, order> const& indices) const -> value_type const&
    {
        auto& self = static_cast<Derived const&>(*this);
        return self.Data()[this->ComputeLinearIndex(indices)];
    }

private:

    inline auto ComputeLinearIndex(std::array<size_t, order> const& indices) const -> size_t
    {
        auto& self = static_cast<Derived const&>(*this);
        if constexpr (has_offset)
        {
            return std::inner_product(indices.begin(), indices.end(), self.Strides().begin(), self.Offset());
        }
        else
        {
            return std::inner_product(indices.begin(), indices.end(), self.Strides().begin(), static_cast<size_t>(0));
        }
    }
};

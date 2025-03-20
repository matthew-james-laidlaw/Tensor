#pragma once

#include "../Containers/Forward.hpp"
#include "../Containers/Traits.hpp"

#include <array>
#include <concepts>
#include <numeric>
#include <tuple>
#include <utility>

struct Range
{
    size_t start;
    size_t stop;
};

template <typename... Slices>
constexpr auto CountRangeSlices() -> size_t
{
    return ((std::same_as<std::decay_t<Slices>, Range> ? 1 : 0) + ...);
}

template <typename T, typename... Slices>
auto SliceImpl(T& tensor, Slices&&... slicePack)
{
    using value_type = TensorTraits<T>::value_type;
    static constexpr size_t order = TensorTraits<T>::order;
    static constexpr bool has_offset = TensorTraits<T>::has_offset;

    constexpr size_t NewOrder = CountRangeSlices<Slices...>();

    size_t offset = 0;
    if constexpr (has_offset)
    {
        offset = tensor.Offset();
    }

    std::array<size_t, NewOrder> newShape{};
    std::array<size_t, NewOrder> newStrides{};
    size_t currentOutputDimension = 0;

    auto slices = std::make_tuple(std::forward<Slices>(slicePack)...);

    auto processDimension = [&](auto i)
    {
        constexpr size_t currentInputDimension = decltype(i)::value;
        auto slice = std::get<i>(slices);

        if constexpr (std::convertible_to<decltype(slice), size_t>)
        {
            size_t sliceIndex = static_cast<size_t>(slice);
            offset += sliceIndex * tensor.Strides()[currentInputDimension];
        }
        else if constexpr (std::same_as<decltype(slice), Range>)
        {
            auto [sliceStart, sliceStop] = slice;
            offset += sliceStart * tensor.Strides()[currentInputDimension];
            newShape[currentOutputDimension] = sliceStop - sliceStart;
            newStrides[currentOutputDimension++] = tensor.Strides()[currentInputDimension];
        }
    };

    auto indexSequence = [&]<std::size_t... I>(std::index_sequence<I...>)
    {
        (processDimension(std::integral_constant<size_t, I>{}), ...);
    };
    indexSequence(std::make_index_sequence<order>{});

    return View<value_type, NewOrder>{tensor.Data(), newShape, newStrides, offset};
}

/*
 * Sliceable mixin for tensor-like classes.
 * Provides multidimensional slicing via the Slice() function.
 */
template <typename Derived, size_t Order>
class Sliceable
{
public:

    template <typename... Slices>
    auto Slice(Slices&&... slices)
    {
        auto& self = static_cast<Derived&>(*this);
        return SliceImpl<Derived>(self, std::forward<Slices>(slices)...);
    }
};

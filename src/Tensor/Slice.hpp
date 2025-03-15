#pragma once

#include "TensorFwd.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <variant>
#include <vector>

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

template <typename T, size_t Order, typename... Slices>
auto SliceImpl(std::array<size_t, Order> shape, std::array<size_t, Order> strides, const T* data, Slices&&... slice_pack)
{
    // index slices reduce dimensionality by one, range slices preserve it (albeit with a constrained range of elements)
    // thus, the remaining output dimensionality is equivalent to the number of range slices provided
    constexpr size_t NewOrder = CountRangeSlices<Slices...>();

    // the resulting view needs a pointer offset, a new shape based on the given slices, and the set of strides that still
    // matter for calculating linear indices non-contiguously
    size_t offset = 0;
    std::array<size_t, NewOrder> new_shape;
    std::array<size_t, NewOrder> new_strides;
    size_t current_output_dimension = 0;

    auto slices = std::make_tuple(slice_pack...);

    auto process_dimension = [&](auto i)
    {
        constexpr size_t current_input_dimension = decltype(i)::value;

        auto slice = std::get<i>(slices);

        if constexpr (std::convertible_to<decltype(slice), size_t>)
        {
            auto slice_index = static_cast<size_t>(slice);

            // advance the offset to the beginning index of this slice
            offset += slice_index * strides[current_input_dimension];
        }
        else if constexpr (std::same_as<decltype(slice), Range>)
        {
            auto [slice_start, slice_stop] = static_cast<Range>(slice);

            // advance the offset to the beginning index of this slice
            offset += slice_start * strides[current_input_dimension];

            // constrain the range of elements to view in this dimension
            new_shape[current_output_dimension] = slice_stop - slice_start;

            // preserve the stride information for this dimension for the resulting view
            new_strides[current_output_dimension++] = strides[current_input_dimension];
        }
    };

    auto index_sequence = [&]<std::size_t... I>(std::index_sequence<I...>)
    {
        (process_dimension(std::integral_constant<size_t, I>{}), ...);
    };
    index_sequence(std::make_index_sequence<Order>{});

    return TensorView<T, NewOrder>{data, new_shape, new_strides, offset};
}

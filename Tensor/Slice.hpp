#pragma once

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

// Assume TensorView is defined elsewhere. For example:
template <typename T, size_t Order>
class TensorView;

/** 
 * @brief Given a parameter pack of slice types, count the
 *        number of 'Range' slices present in the pack.
 */
template <typename... Slices>
constexpr auto CountRangeSlices() -> size_t
{
    return ((std::same_as<std::decay_t<Slices>, Range> ? 1 : 0) + ...);
}

template <typename T, size_t Order, typename... Slices>
auto SliceImpl(std::array<size_t, Order> shape_, std::array<size_t, Order> strides_, std::vector<T> const& data_, Slices&&... slice_pack)
{
	constexpr size_t new_order = CountRangeSlices<Slices...>();
	
	size_t offset = 0;
	std::array<size_t, new_order> new_shape;
	std::array<size_t, new_order> new_strides;
	size_t output_dimension = 0;

    auto slices = std::make_tuple(slice_pack...);

    auto process_dimension = [&](auto i)
    {
        constexpr size_t input_dimension = decltype(i)::value;
        auto slice = std::get<i>(slices);
        
        if constexpr (std::convertible_to<decltype(slice), size_t>)
        {
            auto slice_index = static_cast<size_t>(slice);
            offset += slice_index * strides_[input_dimension];
        }
        else if constexpr (std::same_as<decltype(slice), Range>)
        {
            auto [slice_start, slice_stop] = static_cast<Range>(slice);
            offset += slice_start * strides_[input_dimension];
            new_shape[output_dimension] = slice_stop - slice_start;
            new_strides[output_dimension] = strides_[input_dimension];
            ++output_dimension;
        }
    };

    auto index_sequence = [&]<std::size_t... I>(std::index_sequence<I...>)
    {
        (process_dimension(std::integral_constant<size_t, I>{}), ...);
    };
    index_sequence(std::make_index_sequence<Order>{});

    return TensorView<T, new_order>{ data_.data(), new_shape, new_strides, offset };
}

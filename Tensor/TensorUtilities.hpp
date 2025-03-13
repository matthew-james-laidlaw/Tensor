#pragma once

#include <algorithm>
#include <array>
#include <numeric>
#include <initializer_list>

template <size_t Order>
auto ComputeStrides(std::array<size_t, Order> const& shape) -> std::array<size_t, Order>
{
	auto strides = std::array<size_t, Order>();
	std::transform(shape.rbegin(), shape.rend(), strides.rbegin(),
		[&, product = size_t(1)](size_t dimension) mutable -> size_t
		{
			size_t current = product;
			product *= dimension;
			return current;
		});
	return strides;
}

template <size_t Order>
auto ComputeSize(std::array<size_t, Order> const& shape) -> size_t
{
	return std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<>());
}

template <size_t Order, typename... Ts>
concept ValidShapeOrIndex = (sizeof...(Ts) == Order) && (std::convertible_to<Ts, size_t> && ...);

template <typename T>
concept RawSliceArg = std::same_as<T, size_t> ||
std::same_as<T, std::array<size_t, 0>> ||
std::same_as<T, std::array<size_t, 2>> ||
std::same_as<T, std::initializer_list<size_t>>;

template <size_t Order, typename... Ts>
concept ValidSlices = (sizeof...(Ts) > 0 && sizeof...(Ts) <= Order) &&
(RawSliceArg<Ts> && ...);

constexpr auto convertSlice(auto&& arg)
{
    using Decayed = std::decay_t<decltype(arg)>;
    if constexpr (std::same_as<Decayed, std::initializer_list<size_t>>) {
        // If the initializer list is empty, treat as a full slice.
        if (arg.size() == 0)
            return std::array<size_t, 0>{};
        // If it has two elements, treat it as a partial slice.
        else if (arg.size() == 2) {
            std::array<size_t, 2> arr;
            std::copy(arg.begin(), arg.end(), arr.begin());
            return arr;
        }
        else {
            static_assert(arg.size() == 0 || arg.size() == 2,
                "Initializer list must be empty (for full slice) or have exactly 2 elements (for a partial slice).");
        }
    }
    else {
        // If it's not an initializer list, forward it as-is.
        return std::forward<decltype(arg)>(arg);
    }
}

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
concept IndexSliceType = std::convertible_to<T, size_t>;

template <typename T>
concept RangeOrFullSliceType = requires { typename T::value_type; }
                            && std::same_as<T, std::array<typename T::value_type, 2>>
                            && std::convertible_to<typename T::value_type, size_t>;

template <typename T>
concept SliceType = IndexSliceType<T> || RangeOrFullSliceType<T>;


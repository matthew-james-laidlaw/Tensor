#pragma once

template <typename T, size_t Order>
template <typename... Indices>
	requires(ValidShapeOrIndex<Order, Indices...>)
inline auto Tensor<T, Order>::operator()(Indices... indices) -> T&
{
	return data_[LinearIndex(indices...)];
}

template <typename T, size_t Order>
template <typename... Indices>
	requires(ValidShapeOrIndex<Order, Indices...>)
inline auto Tensor<T, Order>::operator()(Indices... indices) const -> T const&
{
	return data_[LinearIndex(indices...)];
}

template <typename T, size_t Order>
auto Tensor<T, Order>::Size() const -> size_t
{
	return data_.size();
}

template <typename T, size_t Order>
auto Tensor<T, Order>::Dimensions() const -> std::array<size_t, Order> const&
{
	return shape_;
}

template <typename T, size_t Order>
auto Tensor<T, Order>::Data() -> T*
{
	return data_.data();
}

template <typename T, size_t Order>
auto Tensor<T, Order>::Data() const -> const T*
{
	return data_.data();
}

template <typename T, size_t Order>
template <typename... Indices>
	requires(ValidShapeOrIndex<Order, Indices...>)
inline auto Tensor<T, Order>::LinearIndex(Indices... indices) const -> size_t
{
	auto indices_array = std::array<size_t, Order>{ static_cast<size_t>(indices)... };
	return std::inner_product(strides_.begin(), strides_.end(), indices_array.begin(), size_t(0));
}

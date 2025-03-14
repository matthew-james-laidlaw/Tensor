#pragma once

template <typename T, size_t Order>
Tensor<T, Order>::Tensor(std::array<size_t, Order> const& shape)
	: shape_(shape)
	, strides_(ComputeStrides(shape_))
	, data_(ComputeSize(shape_))
{
}

template <typename T, size_t Order>
template <typename... Shape>
	requires(ValidShapeOrIndex<Order, Shape...>)
Tensor<T, Order>::Tensor(Shape... shape)
	: Tensor<T, Order>(std::array<size_t, Order>{ static_cast<size_t>(shape)... })
{
}

template <typename T, size_t Order>
template <typename... Shape>
	requires(ValidShapeOrIndex<Order, Shape...>)
Tensor<T, Order>::Tensor(T const& initializer, Shape... shape)
	: Tensor<T, Order>(std::array<size_t, Order>{ static_cast<size_t>(shape)... })
{
	std::fill(data_.begin(), data_.end(), initializer);
}

template <typename T, size_t Order>
Tensor<T, Order>::Tensor(Tensor const& other)
	: Tensor(other.shape_)
{
	std::copy(other.data_.begin(), other.data_.end(), data_.begin());
}

template <typename T, size_t Order>
Tensor<T, Order>::Tensor(Tensor&& other) noexcept
	: Tensor(other.shape_)
{
	data_ = std::move(other.data_);
}

template <typename T, size_t Order>
auto Tensor<T, Order>::operator=(Tensor const& other) -> Tensor<T, Order>&
{
	if (this != &other)
	{
		shape_ = other.shape_;
		strides_ = other.strides_;
		data_ = other.data_;
	}
	return *this;
}

template <typename T, size_t Order>
auto Tensor<T, Order>::operator=(Tensor&& other) noexcept -> Tensor<T, Order>&
{
	if (this != &other)
	{
		shape_ = other.shape_;
		strides_ = other.strides_;
		data_ = std::move(other.data_);
	}
	return *this;
}

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
template <SliceType S1, SliceType S2>
auto Tensor<T, Order>::Slice(S1 s1, S2 s2)
{
	size_t new_height = shape_[0];
	size_t new_width = shape_[1];
	size_t offset = 0;

	using S1Type = std::decay_t<decltype(s1)>;
	using S2Type = std::decay_t<decltype(s2)>;

	if constexpr (IndexSliceType<S1Type>)
	{
		new_height = 1;
		offset = LinearIndex(s1, 0);
	}
	else if constexpr (RangeOrFullSliceType<S1Type>)
	{
		if (s1[0] != 0 || s1[1] != 0)
		{
			auto [y_start, y_stop] = s1;
			new_height = y_stop - y_start;
			offset = LinearIndex(y_start, 0);
		}
	}

	if constexpr (IndexSliceType<S2Type>)
	{
		new_width = 1;
		offset += s2;
	}
	else if constexpr (RangeOrFullSliceType<S2Type>)
	{
		if (s2[0] != 0 || s2[1] != 0)
		{
			auto [x_start, x_stop] = s2;
			new_width = x_stop - x_start;
			offset += x_start;
		}
	}

	std::array<size_t, 2> new_shape{ new_height, new_width };
	std::array<size_t, 2> new_strides{ strides_[0], strides_[1] };

	return TensorView<T, 2>(data_.data(), new_shape, new_strides, offset);
}

template <typename T, size_t Order>
auto Tensor<T, Order>::Slice(std::array<size_t, 2> const& s1, std::array<size_t, 2> const& s2)
{
	return Slice<std::array<size_t, 2>, std::array<size_t, 2>>(s1, s2);
}

template <typename T, size_t Order>
template <SliceType S1>
auto Tensor<T, Order>::Slice(S1 const& s1, std::array<size_t, 2> const& s2)
{
	return Slice<S1, std::array<size_t, 2>>(s1, s2);
}

template <typename T, size_t Order>
template <SliceType S2>
auto Tensor<T, Order>::Slice(std::array<size_t, 2> const& s1, S2 const& s2)
{
	return Slice<std::array<size_t, 2>, S2>(s1, s2);
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

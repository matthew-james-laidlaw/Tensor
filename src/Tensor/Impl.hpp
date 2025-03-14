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
Tensor<T, Order>::Tensor(TensorView<T, Order> const& view)
    : Tensor(view.Shape())
{
    // Define a recursive lambda that accumulates indices.
    auto assign_recursive = [&](auto&& self, auto... indices) -> void {
        // Base case: when we have indices for all dimensions, assign the element.
        if constexpr (sizeof...(indices) == Order)
        {
            this->operator()(indices...) = view(indices...);
        }
        else
        {
            // Determine the current dimension (number of accumulated indices).
            constexpr size_t dim = sizeof...(indices);
            // Loop over the valid indices for this dimension.
            for (size_t i = 0; i < view.Shape()[dim]; ++i)
            {
                self(self, indices..., i);
            }
        }
    };

    // Kick off the recursion with no indices.
    assign_recursive(assign_recursive);
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

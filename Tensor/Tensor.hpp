#pragma once

#include "TensorUtilities.hpp"

#include <iostream>
#include <vector>

template <typename T, size_t Order>
class Tensor
{
private:

	std::array<size_t, Order> shape_;
	std::array<size_t, Order> strides_;
	std::vector<T> data_;

public:

	Tensor() = delete;
	~Tensor() = default;

	template <typename... Shape>
		requires(ValidShapeOrIndex<Order, Shape...>)
	Tensor(Shape... shape)
		: Tensor(std::array<size_t, Order>{ static_cast<size_t>(shape)... })
	{}

	template <typename... Shape>
		requires(ValidShapeOrIndex<Order, Shape...>)
	Tensor(T const& initializer, Shape... shape)
		: Tensor(std::array<size_t, Order>{ static_cast<size_t>(shape)... })
	{
		std::fill(data_.begin(), data_.end(), initializer);
	}

	Tensor(Tensor const& other)
		: Tensor(other.shape_)
	{
		std::copy(other.data_.begin(), other.data_.end(), data_.begin());
	}

	Tensor(Tensor&& other) noexcept
		: Tensor(other.shape_)
	{
		data_ = std::move(other.data_);
	}

	Tensor& operator=(Tensor const& other)
	{
		if (this != &other)
		{
			shape_ = other.shape_;
			strides_ = other.strides_;
			data_ = other.data_;
		}
		return *this;
	}

	Tensor& operator=(Tensor&& other) noexcept
	{
		if (this != &other)
		{
			shape_ = other.shape_;
			strides_ = other.strides_;
			data_ = std::move(other.data_);
		}
		return *this;
	}

	template <typename... Indices>
		requires(ValidShapeOrIndex<Order, Indices...>)
	inline auto operator()(Indices... indices) -> T&
	{
		return data_[LinearIndex(indices...)];
	}

	template <typename... Indices>
		requires(ValidShapeOrIndex<Order, Indices...>)
	inline auto operator()(Indices... indices) const -> T const&
	{
		return data_[LinearIndex(indices...)];
	}

	friend std::ostream& operator<<(std::ostream& out, Tensor const& tensor)
	{
		out << "Tensor(shape = [";
		for (size_t i = 0; i < tensor.shape_.size(); ++i)
		{
			out << tensor.shape_[i];
			if (i < tensor.shape_.size() - 1)
			{
				out << ", ";
			}
		}
		out << "], data = [";
		for (size_t i = 0; i < tensor.data_.size(); ++i)
		{
			out << tensor.data_[i];
			if (i < tensor.data_.size() - 1)
			{
				out << ", ";
			}
		}
		out << "])";
		return out;
	}

private:

	Tensor(std::array<size_t, Order> const& shape)
		: shape_(shape)
		, strides_(ComputeStrides(shape_))
		, data_(ComputeSize(shape_))
	{}

	template <typename... Indices>
		requires(ValidShapeOrIndex<Order, Indices...>)
	inline auto LinearIndex(Indices... indices) const -> size_t
	{
		auto indices_array = std::array<size_t, Order>{ static_cast<size_t>(indices)... };
		return std::inner_product(strides_.begin(), strides_.end(), indices_array.begin(), size_t(0));
	}

};

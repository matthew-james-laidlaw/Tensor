#pragma once

#include "TensorUtilities.hpp"

#include <iostream>
#include <vector>

template <typename T>
concept SliceArg = std::same_as<T, std::initializer_list<int>>;

template <typename T, size_t Order>
class TensorView;

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

	template <SliceType S1, SliceType S2 = std::array<size_t, 2>>
	auto Slice(S1 s1, S2 s2 = std::array<size_t, 2>{ 0, 0 })
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

	auto Slice(std::array<size_t, 2> const& s1, std::array<size_t, 2> const& s2)
	{
		return Slice<std::array<size_t, 2>, std::array<size_t, 2>>(s1, s2);
	}

	template <SliceType S1>
	auto Slice(S1 const& s1, std::array<size_t, 2> const& s2)
	{
		return Slice<S1, std::array<size_t, 2>>(s1, s2);
	}

	template <SliceType S2 = std::array<size_t, 2>>
	auto Slice(std::array<size_t, 2> const& s1, S2 const& s2 = std::array<size_t, 2>{ 0, 0 })
	{
		return Slice<std::array<size_t, 2>, S2>(s1, s2);
	}

	auto Data() -> T*
	{
		return data_.data();
	}

	auto Data() const -> const T*
	{
		return data_.data();
	}

	auto Size() const -> size_t
	{
		return data_.size();
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

// A simple TensorView class that references the Tensor's data.
template <typename T, size_t Order>
class TensorView {
public:
	TensorView(T* data,
		std::array<size_t, Order> shape,
		std::array<size_t, Order> strides,
		size_t offset)
		: data_(data), shape_(shape), strides_(strides), offset_(offset)
	{
	}

	friend std::ostream& operator<<(std::ostream& out, const TensorView<T, Order>& tv) {
		out << "TensorView(shape = [";
		for (size_t i = 0; i < tv.shape_.size(); ++i) {
			out << tv.shape_[i];
			if (i < tv.shape_.size() - 1) {
				out << ", ";
			}
		}
		out << "], data = [";

		// Compute total number of elements.
		size_t total = 1;
		for (auto s : tv.shape_) {
			total *= s;
		}

		// For each flattened index, compute the corresponding multi-index and then the element offset.
		for (size_t idx = 0; idx < total; ++idx) {
			std::array<size_t, Order> indices{};
			size_t temp = idx;
			// Convert the flat index into multi-index coordinates.
			for (int dim = Order - 1; dim >= 0; --dim) {
				indices[dim] = temp % tv.shape_[dim];
				temp /= tv.shape_[dim];
			}
			// Compute the element offset.
			size_t element_offset = tv.offset_;
			for (size_t d = 0; d < Order; ++d) {
				element_offset += indices[d] * tv.strides_[d];
			}
			out << tv.data_[element_offset];
			if (idx + 1 < total) {
				out << ", ";
			}
		}

		out << "])";
		return out;
	}

private:
	T* data_;
	std::array<size_t, Order> shape_;
	std::array<size_t, Order> strides_;
	size_t offset_;
};


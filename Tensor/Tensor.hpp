#pragma once

#include "Utilities.hpp"

#include <array>
#include <iostream>
#include <vector>

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
	Tensor(Shape... shape);

	template <typename... Shape>
		requires(ValidShapeOrIndex<Order, Shape...>)
	Tensor(T const& initializer, Shape... shape);

	Tensor(TensorView<T, Order> const& view);

	Tensor(Tensor const& other);
	Tensor(Tensor&& other) noexcept;

	auto operator=(Tensor const& other) -> Tensor&;
	auto operator=(Tensor&& other) noexcept -> Tensor&;

	template <typename... Indices>
		requires(ValidShapeOrIndex<Order, Indices...>)
	inline auto operator()(Indices... indices)->T&;

	template <typename... Indices>
		requires(ValidShapeOrIndex<Order, Indices...>)
	inline auto operator()(Indices... indices) const->T const&;

	template <SliceType S1, SliceType S2 = std::array<size_t, 2>>
	auto Slice(S1 s1, S2 s2 = std::array<size_t, 2>{ 0, 0 });

	auto Slice(std::array<size_t, 2> const& s1, std::array<size_t, 2> const& s2);

	template <SliceType S1>
	auto Slice(S1 const& s1, std::array<size_t, 2> const& s2);

	template <SliceType S2 = std::array<size_t, 2>>
	auto Slice(std::array<size_t, 2> const& s1, S2 const& s2 = std::array<size_t, 2>{ 0, 0 });

	auto Size() const -> size_t;
	auto Dimensions() const -> std::array<size_t, Order> const&;
	auto Data() -> T*;
	auto Data() const -> const T*;

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

	Tensor(std::array<size_t, Order> const& shape);

	template <typename... Indices>
		requires(ValidShapeOrIndex<Order, Indices...>)
	inline auto LinearIndex(Indices... indices) const->size_t;

};

#include "View.hpp"
#include "Impl.hpp"

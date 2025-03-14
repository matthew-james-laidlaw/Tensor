template <typename T, size_t Order>
class TensorView
{
public:
	TensorView(const T* data,
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

	auto Shape() const -> std::array<size_t, Order> const&
	{
		return shape_;
	}

	// Indexing operator (const version).
	template <typename... Indices>
		requires (sizeof...(Indices) == Order)
	inline auto operator()(Indices... indices) const -> T const& {
		std::array<size_t, Order> idx{ static_cast<size_t>(indices)... };
		size_t flat_index = offset_;
		for (size_t i = 0; i < Order; ++i) {
			flat_index += idx[i] * strides_[i];
		}
		return data_[flat_index];
	}

private:
	const T* data_;
	std::array<size_t, Order> shape_;
	std::array<size_t, Order> strides_;
	size_t offset_;
};

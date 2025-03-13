#include "Tensor.hpp"

int main()
{
	Tensor<int, 2> matrix(4, 4);

	int i = 0;

	for (size_t y = 0; y < 4; ++y)
	{
		for (size_t x = 0; x < 4; ++x)
		{
			matrix(y, x) = i++;
		}
	}

	std::cout << matrix << std::endl;
	
	auto b1 = matrix.Slice(size_t(2));
	auto b2 = matrix.Slice(std::array<size_t, 2>{0, 2});
	auto b3 = matrix.Slice(std::array<size_t, 0>{});
	auto b4 = matrix.Slice(size_t(2), size_t(2));
	auto b5 = matrix.Slice(size_t(2), std::array<size_t, 2>{0, 2});
	auto b6 = matrix.Slice(std::array<size_t, 2>{0, 4}, std::array<size_t, 2>{0, 4});

	std::cout << b1 << std::endl;
	std::cout << b2 << std::endl;
	std::cout << b3 << std::endl;
	std::cout << b4 << std::endl;
	std::cout << b5 << std::endl;
	std::cout << b6 << std::endl;

	return 0;
}

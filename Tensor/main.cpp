#include "Tensor.hpp"

#include <array>
#include <iostream>

int main()
{
	Tensor<int, 2> matrix(4, 4);
	std::iota(matrix.Data(), matrix.Data() + matrix.Size(), 0);

	auto s1 = matrix.Slice(1, 1);
	auto s2 = matrix.Slice(1, Range{ 1, 3 });
	//auto s3 = matrix.Slice({ 1, 3 }, 3);
	auto s4 = matrix.Slice(Range{ 1, 3 }, Range{ 2, 4 });
	//auto s5 = matrix.Slice({ 1 });
	//auto s6 = matrix.Slice({ 1, 3 });

	std::cout << matrix << std::endl;
	std::cout << s1 << std::endl;
	std::cout << s4 << std::endl;

	return 0;
}

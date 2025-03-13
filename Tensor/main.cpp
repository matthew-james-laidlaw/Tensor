#include "Tensor.hpp"

#include <array>
#include <iostream>

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

	auto s1 = matrix.Slice(1, 3);				 // index, index
	auto s2 = matrix.Slice(1, { 1, 3 });		 // index, range
	auto s3 = matrix.Slice(1, {});			     // index, full
	auto s4 = matrix.Slice({ 1, 3 }, 3);		 // range, index
	auto s5 = matrix.Slice({ 1, 3 }, { 2, 4 });  // range, range
	auto s6 = matrix.Slice({ 1, 3 }, {});		 // range, full
	auto s7 = matrix.Slice({}, 1);			     // full,  index
	auto s8 = matrix.Slice({}, { 1, 3 });		 // full,  range
	auto s9 = matrix.Slice({}, {});			     // full,  full
	auto s10 = matrix.Slice(1);					 // index, implicit-full
	auto s11 = matrix.Slice({ 1, 3 });			 // range, implicit-full
	auto s12 = matrix.Slice({});				 // full,  implicit-full

	std::cout << s1 << std::endl;
	std::cout << s2 << std::endl;
	std::cout << s3 << std::endl;
	std::cout << s4 << std::endl;
	std::cout << s5 << std::endl;
	std::cout << s6 << std::endl;
	std::cout << s7 << std::endl;
	std::cout << s8 << std::endl;
	std::cout << s9 << std::endl;
	std::cout << s10 << std::endl;
	std::cout << s11 << std::endl;
	std::cout << s12 << std::endl;

	return 0;
}

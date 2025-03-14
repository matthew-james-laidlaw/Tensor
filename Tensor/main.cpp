#include "Tensor.hpp"

#include <array>
#include <iostream>

int main()
{
	Tensor<int, 2> matrix(4, 4);
	std::iota(matrix.Data(), matrix.Data() + matrix.Size(), 0);

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

	Tensor<int, 2> tt(s11);
	std::cout << tt << std::endl;

	return 0;

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

	Tensor<int, 2> t1(4, 4);
	std::iota(t1.Data(), t1.Data() + t1.Size(), 0);

	auto t2(t1);
	std::cout << t2 << std::endl;

	auto t3(std::move(t2));
	std::cout << t3 << std::endl;

	auto t4 = t1;
	std::cout << t4 << std::endl;

	auto t5 = std::move(t4);
	std::cout << t5 << std::endl;

	return 0;
}

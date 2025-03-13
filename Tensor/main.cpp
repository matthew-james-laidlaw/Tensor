#include "Tensor.hpp"

int main()
{
	Tensor<size_t, 2> m1(2, 2);
	Tensor<size_t, 2> m2(42, 2, 2);

	std::cout << m1 << std::endl;
	std::cout << m2 << std::endl;
}

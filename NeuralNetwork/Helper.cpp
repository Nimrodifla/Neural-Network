#include "Helper.h"

int rndNum = 13;

float Helper::definedValue(float x)
{
	if (x < 0)
	{
		return -x;
	}
	return x;
}

float Helper::randomFloat()
{
	std::srand(std::time(nullptr));
	return (std::rand() + rndNum);
}
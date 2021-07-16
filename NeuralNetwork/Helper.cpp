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

std::vector<float> Helper::vectorSub(std::vector<float> a, std::vector<float> b)
{
	int i = 0;

	std::vector<float> result;

	if (a.size() != b.size())
	{
		throw std::exception("Vectors aren't the same length!");
	}

	for (i = 0; i < a.size(); i++)
	{
		result.push_back(a[i] - b[i]);
	}

	return result;
}

float Helper::randomFloatRange(float low, float high)
{
	if (high - low == 0)
	{
		return 0;
	}

	std::srand(time(0));
	float r = low + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (high - low)));

	return r;
}

float Helper::scaleBetweenZeroAndOne(float num)
{
	float a = (1 / ((float)WHEN_FORMULA_IS_ONE));

	float temp = ((a * definedValue(num)) + (a * num)) / 2;

	if (temp > 1)
	{
		return (float)1.0;
	}

	return temp;
}
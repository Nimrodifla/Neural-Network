#include "Helper.h"

int rndNum = 13 + time(nullptr);

float Helper::definedValue(float x)
{
	if (x < 0)
	{
		return (-1 * x);
	}
	return x;
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

std::vector<float> Helper::vectorAdd(std::vector<float> a, std::vector<float> b)
{
	int i = 0;

	std::vector<float> result;

	if (a.size() != b.size())
	{
		throw std::exception("Vectors aren't the same length!");
	}

	for (i = 0; i < a.size(); i++)
	{
		result.push_back(a[i] + b[i]);
	}

	return result;
}

float Helper::randomFloatRange(float low, float high)
{
	if (high - low == 0)
	{
		return 0;
	}

	std::srand(rndNum);
	float num = rand();
	rndNum += 1371;
	rndNum = rndNum % RAND_MAX;
	float r = low + static_cast <float> (num) / (static_cast <float> (RAND_MAX / (high - low)));

	return r;
}

float Helper::scaleBetweenZeroAndOne(float num)
{
	float temp = 1 / (1 + powf(1.5, -num));

	return temp;
}

float Helper::generationBasedNudge(int gen)
{
	float max = (gen / 100) / powf((gen / 100), 2);
	if (max > MAX_NUDGE)
	{
		max = MAX_NUDGE;
	}

	float temp = randomFloatRange(0, max); // is just random, not based on the gen

	return temp;
}

void Helper::addLineToFile(std::ofstream& file, std::string line)
{
	file << line << "\n";
}
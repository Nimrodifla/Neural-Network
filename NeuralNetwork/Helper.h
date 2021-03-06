#pragma once
#include <string>
#include <vector>
#include <ctime>
#include <fstream>

#define WHEN_FORMULA_IS_ONE 5

class Helper
{

public:
	static float definedValue(float x);
	static std::vector<float> vectorSub(std::vector<float> a, std::vector<float> b);
	static std::vector<float> vectorAdd(std::vector<float> a, std::vector<float> b);
	static float randomFloatRange(float low, float high);
	static float scaleBetweenZeroAndOne(float num);
	static float generationBasedNudge(int gen);
	static void addLineToFile(std::ofstream& file, std::string line);

	template <typename T>
	static std::vector<T> vectorClone(std::vector<T> vectorToClone)
	{
		int i = 0;

		std::vector<T> result;

		for (i = 0; i < vectorToClone.size(); i++)
		{
			result.push_back(vectorToClone[i]);
		}

		return result;
	}
};
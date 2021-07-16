#pragma once
#include <string>
#include <vector>
#include <ctime>

class Helper
{
public:
	static float definedValue(float x);
	static float randomFloat();
	static std::vector<float> vectorSub(std::vector<float> a, std::vector<float> b);
	static float randomFloatRange(float low, float high);
};
#pragma once
#include <string>
#include <vector>
#include <ctime>
#include <fstream>
#include <iostream>

#define WHEN_FORMULA_IS_ONE 5
#define MAX_NUDGE 2

struct NeuronPos
{
	int layerIn;
	int neuronIn;
} typedef NeuronPos;

struct Matrix
{
	int rows;
	int colums;
	float** maxrix;
} typedef Matrix;

class Helper
{

public:
	static float definedValue(float x);
	static std::vector<float> vectorSub(std::vector<float> a, std::vector<float> b);
	static std::vector<float> vectorAdd(std::vector<float> a, std::vector<float> b);
	static float randomFloatRange(float low, float high);
	static float scaleBetweenZeroAndOne(float num);
	static float ReLU(float x);
	static Matrix matrixReLU(Matrix mat);
	static float generationBasedNudge(int gen);
	static void addLineToFile(std::ofstream& file, std::string line);
	static Matrix matrixMultiplication(Matrix a, Matrix b);
	static void rotateMatrix(Matrix matrix);
	static float matrixSum(Matrix mat);
	static int deriv_ReLU(float x);
	static Matrix matrixDerinReLU(Matrix mat);
	static float** randomMaxrixInRange(int r, int c, float min, float max);
	static void freeMatrix(Matrix mat);
	static float* softmax(float* arr, int size);
	static Matrix matrixSoftmax(Matrix mat);
	static Matrix cloneMatrix(Matrix mat);
	static void printMatrix(Matrix mat);
	static void multiMatrixBy(Matrix mat, float a);

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
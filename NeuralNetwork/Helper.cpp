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

float Helper::ReLU(float x)
{
	float temp = (fabsf(x) + x) / 2.0;

	return temp;
}

float Helper::generationBasedNudge(int gen)
{
	/*
	float max = (gen / 100.0) / powf((gen / 100.0), 2.0);
	if (max > MAX_NUDGE)
	{
		max = MAX_NUDGE;
	}
	*/
	float temp = randomFloatRange(0.1, 1); // is just random, not based on the gen

	return temp;
}

void Helper::addLineToFile(std::ofstream& file, std::string line)
{
	file << line << "\n";
}

int Helper::deriv_ReLU(float x)
{
	return x > 0;
}

float** Helper::randomMaxrixInRange(int r, int c, float min, float max)
{
	int i = 0, j = 0;

	float** mat = new float* [r];
	for (i = 0; i < r; i++)
	{
		mat[i] = new float[c];
		for (j = 0; j < c; j++)
		{
			mat[i][j] = Helper::randomFloatRange(min, max);
		}
	}

	return mat;
}

void Helper::freeMatrix(Matrix mat)
{
	int i = 0;

	for (i = 0; i < mat.rows; i++)
	{
		delete[] mat.maxrix[i];
	}

	delete[] mat.maxrix;
}

float* Helper::softmax(float* arr, int size)
{
	int i = 0;

	float sum = 0;

	for (i = 0; i < size; i++)
	{
		sum += arr[i];
	}

	float* result = new float[size];
	for (i = 0; i < size; i++)
	{
		result[i] = arr[i] / sum;
	}

	return result;
}

Matrix Helper::matrixSoftmax(Matrix mat)
{
	int i = 0;
	Matrix result;
	result.rows = mat.rows;
	result.colums = mat.colums;
	result.maxrix = new float* [result.rows];
	for (i = 0; i < result.rows; i++)
	{
		result.maxrix[i] = Helper::softmax(mat.maxrix[i], mat.colums);
	}

	return result;
}

Matrix Helper::matrixReLU(Matrix mat)
{
	int i = 0, j = 0;
	Matrix result;
	result.rows = mat.rows;
	result.colums = mat.colums;
	result.maxrix = new float* [result.rows];
	
	for (i = 0; i < result.rows; i++)
	{
		result.maxrix[i] = new float[result.colums];
		for (j = 0; j < result.colums; j++)
		{
			result.maxrix[i][j] = Helper::ReLU(mat.maxrix[i][j]);
		}
	}

	return result;
}

Matrix Helper::matrixMultiplication(Matrix a, Matrix b)
{
	int u = 0, i = 0, j = 0;

	Matrix c;
	// init c
	c.rows = a.rows;
	c.colums = b.colums;
	c.maxrix = new float* [c.rows];
	for (i = 0; i < c.rows; i++)
	{
		c.maxrix[i] = new float[c.colums];
	}

	for (i = 0; i < a.rows; i++) {
		for (j = 0; j < b.colums; j++) {
			c.maxrix[i][j] = 0;
			for (u = 0; u < a.rows; u++) {
				c.maxrix[i][j] += a.maxrix[i][u] * b.maxrix[u][j];
			}
		}
	}

	return c;
}

Matrix Helper::cloneMatrix(Matrix mat)
{
	int i = 0, j = 0;

	Matrix result;
	result.rows = mat.rows;
	result.colums = mat.colums;
	result.maxrix = new float* [result.rows];

	for (i = 0; i < result.rows; i++)
	{
		result.maxrix[i] = new float[result.colums];
		for (j = 0; j < result.colums; j++)
		{
			result.maxrix[i][j] = mat.maxrix[i][j];
		}
	}

	return result;
}

void Helper::printMatrix(Matrix mat)
{
	int i = 0, j = 0;
	for (i = 0; i < mat.rows; i++)
	{
		for (j = 0; j < mat.colums; j++)
		{
			std::cout << mat.maxrix[i][j] << " ";
		}
		std::cout << "\n";
	}
}

Matrix Helper::matrixDerinReLU(Matrix mat)
{
	int i = 0, j = 0;

	Matrix result;
	result.rows = mat.rows;
	result.colums = mat.colums;
	result.maxrix = new float* [result.rows];

	for (i = 0; i < result.rows; i++)
	{
		result.maxrix[i] = new float[result.colums];
		for (j = 0; j < result.colums; j++)
		{
			result.maxrix[i][j] = Helper::deriv_ReLU(mat.maxrix[i][j]);
		}
	}

	return result;
}

void Helper::multiMatrixBy(Matrix mat, float a)
{
	int i = 0, j = 0;

	for (i = 0; i < mat.rows; i++)
	{
		for (j = 0; j < mat.colums; j++)
		{
			mat.maxrix[i][i] *= a;
		}
	}
}
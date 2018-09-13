#ifndef __COMMON_H__
#define __COMMON_H__

#include "config.h" // cmake 编译控制 

// cuda 运行时函数
#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
// cuda API error message
#define CHECK(x) \
do {\
	cudaError_t err = (x);\
	if (err != cudaSuccess) {\
		 fprintf(stderr, "API error %s:%d Returned:%s\n", __FILE__, __LINE__, cudaGetErrorString(err));\
		 exit(1);\
	}\
} while(0)

#define CHECK_KERNEL() \
do {\
	cudaError_t err = cudaPeekAtLastError();\
	if (err != cudaSuccess) {\
		 fprintf(stderr, "Kernel error %s:%d Returned:%s\n", __FILE__, __LINE__, cudaGetErrorString(err));\
		 exit(1);\
	}\
} while(0)
#endif


#include <vector>
#include <iostream>
#include <cstdio>
#include <cassert>
#include <utility>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <algorithm>

using namespace std;


#define FOR(i, l, r) for(int (i) = (l); (i) <= (r); (i)++)
#define FORS(i, l, r, s) for(int (i) = (l); (i) <= (r); (i) += (s))

// 通用工具类
class Tool {
	public:
	int RandInt(int a, int b); 
	double RandDouble(double a, double b, double eps);
};

extern Tool Tools;

// 如果启用cuda的话， 这些函数需要在device上重新实现
#ifndef ENABLE_CUDA
//激活函数集合
namespace ActiveFunction{
	double Sigmoid(double x);
	double SigmoidDel(double x);

	double ReLU(double x);
	double ReLUDel(double x);

	double tanh(double x);
	double tanhDel(double x);

	double BNLL(double x);
	double BNLLDel(double x);

	double Linear(double x);
	double LinearDel(double x);
};
#endif

#endif


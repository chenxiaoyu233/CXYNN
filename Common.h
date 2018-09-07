#ifndef __COMMON_H__
#define __COMMON_H__

//#define NDEBUG //调试完成之后使用这句话禁用assert

// cuda 运行时函数
#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

// cuda kernels
#ifdef ENABLE_CUDA
#include "cuda/kernels.h"
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


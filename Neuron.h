#ifndef __NEURON_H__
#define __NEURON_H__

#include "Common.h"
#include "Fiber.h"

// 用于区分树突和轴突的枚举
enum NeuronIO { INPUT = 0, OUTPUT = 1 };

// 解锁
class Fiber;

class Neuron {
	public: //先把全部访问权限开放
	double (*ActiveFunc) (double); //非线性变换函数
	double (*ActiveFuncDelta) (double); //非线性变换函数的导函数

	double forwardBuffer[3]; //模拟网络行为时使用的临时存储(正常运行)
	// 0 x, 1 ActiveFunc(x), 2 ActiveFuncDelta(x)

	double backwardBuffer; //使用反向传播算法的时候使用的存储
	// 总的输出对激活函数内部的偏导数

	vector<Fiber> input; //输入边(树突)
	vector<Fiber> output; //输出边(轴突)

	double* b; //偏移
	double* bDel; //偏移量的偏导数

	bool dropOutFlag; //留用, 防止过拟合.

	Neuron(); 
	Neuron(double (*ActiveFunc) (double), 
		   double (*ActiveFuncDelta) (double), 
		   double* b, double* bDel);

	void Insert(NeuronIO type, double* weight, double* dweight, Neuron* neighbor);
	void SetValue(double x); //设置forwardBuffer[0]
	void SetActionFunc(double (*ActiveFunc) (double),
					   double (*ActiveFuncDelta) (double));

	void UpdateBuffer();
	void SpreadBack();

	void Log();
};

#endif

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

#ifdef ENABLE_CUDA
	// 优化 CPU -> GPU 传输的数据量
	vector<Fiber> *input;
	vector<Fiber> *output;

	Fiber *gpu_input, *gpu_output; //在显存中开辟的空间
	Fiber *cpu_input, *cpu_output; //在内存中开辟的空间
	int *gpu_input_idx, *gpu_output_idx;
	int *cpu_input_idx, *cpu_output_idx;
	int *gpu_input_count, *gpu_output_count;
	int *cpu_input_count, *cpu_output_count;
	int idx; // 用于在显存中重建网络
#else
	vector<Fiber> input; //输入边(树突)
	vector<Fiber> output; //输出边(轴突)
#endif

	double* b; //偏移
	double* bDel; //偏移量的偏导数

	bool dropOutFlag; //留用, 防止过拟合.

	Neuron(); 
	Neuron(double (*ActiveFunc) (double), 
		   double (*ActiveFuncDelta) (double), 
		   double* b, double* bDel);

#ifdef ENABLE_CUDA
	~Neuron();
#endif

	void Insert(NeuronIO type, double* weight, double* dweight, Neuron* neighbor);
#ifndef ENABLE_CUDA
	// 启用cuda 之后不能使用这个接口
	void SetValue(double x); //设置forwardBuffer[0]
#endif
	void SetActionFunc(double (*ActiveFunc) (double),
					   double (*ActiveFuncDelta) (double));

#ifndef ENABLE_CUDA
	//如果没有开启cuda则使用这套接口, 使用CPU分别完成
	//每个神经元上的计算
	void UpdateBuffer();
	void SpreadBack();
#endif

#ifdef ENABLE_CUDA
	void SyncFiberInfo();
	void syncFiberInfo(
		vector<Fiber> *vec,
		Fiber **fiber, Fiber **gpu_fiber, 
		int **cpu_count, int **gpu_count,
		int **cpu_idx, int **gpu_idx
	);
#endif
	void Log();
};

#endif

#ifndef __FUNC_ABSTRACTOR_H__
#define __FUNC_ABSTRACTOR_H__

#include "Common.h"
#include "Layer.h"
#include "Matrix.h"
#include "Estimator.h"

extern Tool Tools;

class FuncAbstractor {
	private:
	double loss; // loss其实就是真个函数真正的输出`
	void ClearParamDel();

	double L2param;
	void L2regularization();

	public:
	Estimator* estimator; // 用于最终将网络封装成函数和求一部分参数的偏导数
	Layer *Input, *Output; //暂时公有
	vector<double*> param;
	vector<double*> paramDel;
#ifdef ENABLE_CUDA
	int param_cnt;
	double *cpu_param, *cpu_paramDel; // 在host中暂存参数(非真实参数)
	double *gpu_param, *gpu_paramDel; // 在gpu中暂存参数(非真实参数)
	double **cpu_param_ptr, **cpu_paramDel_ptr; //cpu中指向真实参数的指针(预处理完马上销毁)
	double **gpu_param_ptr, **gpu_paramDel_ptr; //gpu中指向真实参数的指针
#endif

	FuncAbstractor(
		Layer* Input, 
		Layer* Output,
	   	Estimator* estimator,
	   	double L2param
	);
#ifdef ENABLE_CUDA
	~FuncAbstractor();
	void syncParamFromHostToDevice(); // 将cpu_param中的参数同步到GPU中的真实参数
	void syncParamFromDeviceToHost(); // 将GPU中的真实参数同步到cpu_param中.
#endif

	void Randomization(int seed, int l, int r);
	void Randomization(int seed, double l, double r, double eps);
	void Update(Matrix<double> *dataIn, Matrix<double> *expect);
	double GetLoss(); //调用Update方法之后才能得到当前的 loss 
};

#endif

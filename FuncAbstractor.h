#ifndef __FUNC_ABSTRACTOR_H__
#define __FUNC_ABSTRACTOR_H__

#include "Common.h"
#include "Layer.h"
#include "Matrix.h"
#include "Estimator.h"

extern Tool Tools;

class FuncAbstractor {
	private:
	Estimator* estimator; // 用于最终将网络封装成函数和求一部分参数的偏导数
	double loss; // loss其实就是真个函数真正的输出`
	void ClearParamDel();

	double L2param;
	void L2regularization();

	public:
	Layer *Input, *Output; //暂时公有
	vector<double*> param;
	vector<double*> paramDel;

	FuncAbstractor(
		Layer* Input, 
		Layer* Output,
	   	Estimator* estimator,
	   	double L2param
	);

	void Randomization(int seed, int l, int r);
	void Randomization(int seed, double l, double r, double eps);
	void Update(Matrix<double> *dataIn, Matrix<double> *expect);
	double GetLoss(); //调用Update方法之后才能得到当前的 loss 
};

#endif

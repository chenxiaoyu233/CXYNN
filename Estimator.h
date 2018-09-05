#ifndef __ESTIMATOR_H__
#define __ESTIMATOR_H__

#include "Common.h"
#include "Matrix.h"
#include "Layer.h"


class Estimator {
	protected:
	Layer* Input; //网络输出层
	double loss;
	int row, col;

	public:
	Estimator(Layer* Input);
	virtual double Loss(Matrix<double> *expect) = 0; //计算loss
	virtual double LossDel(Matrix<double> *expect) = 0; //计算backwardBuffer
};


//平方损失 f = \sum_{i=1}^n (\theta(x_i) - a_i)^2
class Estimator_QuadraticCost: public Estimator {
	public: 
	Estimator_QuadraticCost(Layer* Input);

	virtual double Loss(Matrix<double> *expect);
	virtual double LossDel(Matrix<double> *expect);
};

// softmax 损失函数
class Estimator_Softmax: public Estimator {
	public: 
	Estimator_Softmax(Layer *Input);

	// expect 为1 x 1的矩阵, 表示预期出现的编号
	virtual double Loss(Matrix<double> *expect);
	virtual double LossDel(Matrix<double> *expect);
};

#endif

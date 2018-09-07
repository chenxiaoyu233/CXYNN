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
	virtual double _loss(Matrix<double> *expect) = 0;
	virtual void _lossDel(Matrix<double> *expect) = 0;

	public:
	Estimator(Layer* Input);
	double Loss(Matrix<double> *expect); //计算loss
	void LossDel(Matrix<double> *expect); //计算backwardBuffer
};


//平方损失 f = \sum_{i=1}^n (\theta(x_i) - a_i)^2
class Estimator_QuadraticCost: public Estimator {
	public: 
	Estimator_QuadraticCost(Layer* Input);

	virtual double _loss(Matrix<double> *expect);
	virtual void _lossDel(Matrix<double> *expect);
};

// softmax 损失函数
class Estimator_Softmax: public Estimator {
	public: 
	Estimator_Softmax(Layer *Input);

	// expect 为1 x 1的矩阵, 表示预期出现的编号
	virtual double _loss(Matrix<double> *expect);
	virtual void _lossDel(Matrix<double> *expect);
};

#endif

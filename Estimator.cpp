#include "Estimator.h"

Estimator::Estimator(Layer* Input) { 
	this -> Input = Input;
	loss = 0;
	pair<int, int> sz = Input -> size();
	row = sz.first; col = sz.second;
}

// Estimator_QuadraticCost

Estimator_QuadraticCost::Estimator_QuadraticCost(Layer* Input):Estimator(Input) { }

double Estimator_QuadraticCost::Loss(Matrix<double> *expect) {
	loss = 0;
	FOR(x, 1, row) FOR(y, 1, col) {
		Neuron& cur = (*Input)(x, y);
		double del = (*expect)(x, y) - cur.forwardBuffer[1];
		loss += del * del;
	}
	return loss;
}

double Estimator_QuadraticCost::LossDel(Matrix<double> *expect) {
	FOR(x, 1, row) FOR(y, 1, col) {
		Neuron& cur = (*Input)(x, y);
		cur.backwardBuffer = 2 * (cur.forwardBuffer[1] - (*expect)(x, y) ) * cur.forwardBuffer[2];
		*cur.bDel = cur.backwardBuffer;
	}
}

// Estimator_Softmax

Estimator_Softmax::Estimator_Softmax(Layer *Input):Estimator(Input){ 
	assert(row == 1); // 保证输入为1行
	// 需要保证最后一层的输出是linear函数
}

double Estimator_Softmax::Loss(Matrix<double> *expect) {
	pair<int, int> sz = expect -> size();
	assert(sz.first == 1); // 1 row
	assert(sz.second == 1); // 1 col

	int idx = int( (*expect)(1) );
	double tot = 0;
	FOR(i, 1, col) tot += exp( (*Input)(i).forwardBuffer[1] );
	return log(tot) - (*Input)(idx).forwardBuffer[1];
}

double Estimator_Softmax::LossDel(Matrix<double> *expect) {
	pair<int, int> sz = expect -> size();
	assert(sz.first == 1); // 1 row
	assert(sz.second == 1); // 1 col

	int idx = int( (*expect)(1) );
	double tot = 0;
	FOR(i, 1, col) tot += exp( (*Input)(i).forwardBuffer[1] );
	FOR(i, 1, col) { // 计算每一个输入神经元的偏导数
		double f = exp( (*Input)(i).forwardBuffer[1] ) / tot;
		if(i == idx) {
			(*Input)(i).backwardBuffer = (f - 1) * (*Input)(i).forwardBuffer[2];
		} else {
			(*Input)(i).backwardBuffer = f * (*Input)(i).forwardBuffer[2];
		}
		*((*Input)(i).bDel) = (*Input)(i).backwardBuffer;
	}
}

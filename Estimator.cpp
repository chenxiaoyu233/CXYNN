#include "Estimator.h"

Estimator::Estimator(Layer* Input) { 
	this -> Input = Input;
	loss = 0;
	pair<int, int> sz = Input -> size();
	row = sz.first; col = sz.second;
}

double Estimator::Loss(Matrix<double> *expect) {
#ifdef ENABLE_CUDA
	Input -> syncMemFromDeviceToHost();
#endif
	double ret = this -> loss(expect);
#ifdef ENABLE_CUDA
	Input -> syncMemFromHostToDevice(); // 这个可能可以省略
#endif
	return ret;
}

void Estimator::LossDel(Matrix<double> *expect) {
#ifdef ENABLE_CUDA
	Input -> syncMemFromDeviceToHost();
#endif
	this -> lossDel(expect);
#ifdef ENABLE_CUDA
	Input -> syncMemFromHostToDevice();
	Input -> Sync_BackwardBuffer_to_bDel(); // *bDel <= backwardBuffer
#endif
}

// Estimator_QuadraticCost

Estimator_QuadraticCost::Estimator_QuadraticCost(Layer* Input):Estimator(Input) { }

double Estimator_QuadraticCost::loss(Matrix<double> *expect) {
	loss = 0;
	FOR(x, 1, row) FOR(y, 1, col) {
		Neuron& cur = (*Input)(x, y);
		double del = (*expect)(x, y) - cur.forwardBuffer[1];
		loss += del * del;
	}
	return loss;
}

double Estimator_QuadraticCost::lossDel(Matrix<double> *expect) {
	FOR(x, 1, row) FOR(y, 1, col) {
		Neuron& cur = (*Input)(x, y);
		cur.backwardBuffer = 2 * (cur.forwardBuffer[1] - (*expect)(x, y) ) * cur.forwardBuffer[2];
#ifndef ENABLE_CUDA
		*cur.bDel = cur.backwardBuffer;
#endif
	}
}

// Estimator_Softmax

Estimator_Softmax::Estimator_Softmax(Layer *Input):Estimator(Input){ 
	assert(row == 1); // 保证输入为1行
	// 需要保证最后一层的输出是linear函数
}

double Estimator_Softmax::loss(Matrix<double> *expect) {
	pair<int, int> sz = expect -> size();
	assert(sz.first == 1); // 1 row
	assert(sz.second == 1); // 1 col

	int idx = int( (*expect)(1) );
	double tot = 0;
	FOR(i, 1, col) tot += exp( (*Input)(i).forwardBuffer[1] );
	return log(tot) - (*Input)(idx).forwardBuffer[1];
}

double Estimator_Softmax::lossDel(Matrix<double> *expect) {
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
#ifndef ENABLE_CUDA
		*((*Input)(i).bDel) = (*Input)(i).backwardBuffer;
#endif
	}
}

#include "Estimator.h"

Estimator::Estimator(Layer* Input) { 
	this -> Input = Input;
	loss = 0;
	pair<int, int> sz = Input -> size();
	row = sz.first; col = sz.second;
}

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

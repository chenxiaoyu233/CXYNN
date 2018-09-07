#include "Common.h"

Tool Tools;

int Tool::RandInt(int a, int b) {
	int del = b - a + 1;
	return a + rand() % del;
}

double Tool::RandDouble(double a, double b, double eps){
	a /= eps; b /= eps;
	return RandInt(a, b) * eps;
}

#ifndef ENABLE_CUDA
double ActiveFunction::Sigmoid(double x) {
	return 1.0f/(1.0f + exp(-x));
}

double ActiveFunction::SigmoidDel(double x) {
	return Sigmoid(x) * (1 - Sigmoid(x));
}

double ActiveFunction::ReLU(double x) {
	return x < 0 ? 0 : x;
}

double ActiveFunction::ReLUDel(double x) {
	if (x <= 0) return 0;
	if (x > 0) return 1;
}

double ActiveFunction::tanh(double x) {
	return (exp(2*x) - 1) / (exp(2*x) + 1);
}

double ActiveFunction::tanhDel(double x) {
	return 1.0f - tanh(x) * tanh(x);
}

double ActiveFunction::BNLL(double x) {
	return log(1.0f + exp(x));
}

double ActiveFunction::BNLLDel(double x) {
	return exp(x) / (1.0f + exp(x));
}

double ActiveFunction::Linear(double x) {
	return x;
}

double ActiveFunction::LinearDel(double x) {
	return 1.0f;
}
#endif

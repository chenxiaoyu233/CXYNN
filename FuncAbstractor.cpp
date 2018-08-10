#include "FuncAbstractor.h"

FuncAbstractor::FuncAbstractor(Layer* Input, Layer* Output, Estimator* estimator):
	Input(Input), Output(Output), estimator(estimator) {
	param.clear();
	paramDel.clear();
	Output -> CollectParam(&(this -> param), &(this -> paramDel));
	loss = 0;
}

void FuncAbstractor::Randomization(int seed, int l, int r) {
	srand(seed);
	for (int i = 0; i < param.size(); i++) {
		*(param[i]) = Tools.RandInt(l, r);
	}
}

void FuncAbstractor::Randomization(int seed, double l, double r, double eps) {
	srand(seed);
	for (int i = 0; i < param.size(); i++) {
		*(param[i]) = Tools.RandDouble(l, r, eps);
	}
}

void FuncAbstractor::ClearParamDel() {
	for (int i = 0; i < paramDel.size(); i++) *(paramDel[i]) = 0;
}

void FuncAbstractor::Update(Matrix<double> *dataIn, Matrix<double> *expect) {
	ClearParamDel();
	Input -> UpdateForwardBegin(dataIn); // 前向更新值
	loss = estimator -> Loss(expect); //计算loss
	estimator -> LossDel(expect); //更新输出层的backwardBuffer
	Output -> SpreadBack(); //反向传播
}

double FuncAbstractor::GetLoss() {
	return loss;
}

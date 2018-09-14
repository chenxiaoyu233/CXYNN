#include "Predictor.h"

Predictor::Predictor(
	FuncAbstractor* func, 
	double step, 
	int maxEpoch,
	vector<Matrix<double>*> trainData,
	vector<Matrix<double>*> trainLabel,
	string filePath,
	int seed,
	double randL, double randR, double randEps,
	int miniBatchSize

): Optimizer(func, step, maxEpoch, trainData, trainLabel, filePath, 
	seed, randL, randR, randEps, miniBatchSize) {
	expectPair.clear();
	LoadFromFile();
}

int Predictor::Classify(Matrix<double>* Input) {
	int whe = 0;
	double minLoss = 1e10; //标兵, 极限数据可能会出问题

	resetDropoutLayer();

	for (int i = 0; i < expectPair.size(); i++) {
		func -> Update(Input, expectPair[i].second);
		double curLoss = func -> GetLoss();
		if (curLoss < minLoss) {
			whe = i; minLoss = curLoss;
		}
	}
	return expectPair[whe].first;
}

double Predictor::Loss(Matrix<double>* Input, Matrix<double>* expect) {
	func -> Update(Input, expect);
	return func -> GetLoss();
}

void Predictor::AddCase(int label, Matrix<double>* expect) {
	expectPair.push_back(make_pair(label, expect));
}


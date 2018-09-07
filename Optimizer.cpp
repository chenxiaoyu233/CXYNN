#include "Optimizer.h"

Optimizer::Optimizer(
	FuncAbstractor* func, 
	double step, 
	int maxEpoch,
	vector<Matrix<double>*> trainData,
	vector<Matrix<double>*> trainLabel,
	string filePath,
	int seed,
	double randL, double randR, double randEps,
	int miniBatchSize

):  func(func),
    step(step),  
	maxEpoch(maxEpoch), 
	trainData(trainData), 
	trainLabel(trainLabel),
	filePath(filePath),
	seed(seed),
	randL(randL), randR(randR), randEps(randEps),
	miniBatchSize(miniBatchSize){

	epoch = 0;
	saveStep = 100;
	direction.clear();
	direction.resize((func->param).size(), 0);
}

void Optimizer::NormalizeData(Matrix<double>* data) {
	pair<int, int> sz = data -> size();
	double mean = 0; //平均数
	double del = 0;  //模长
	FOR(x, 1, sz.first) FOR(y, 1, sz.second) mean += (*data)(x, y);
	mean /= (sz.first * sz.second);
	FOR(x, 1, sz.first) FOR(y, 1, sz.second) {
		(*data)(x, y) -= mean;
		del += (*data)(x, y) * (*data)(x, y);
	}
	del = sqrt(del);
	FOR(x, 1, sz.first) FOR(y, 1, sz.second) (*data)(x, y) /= del;
}

void Optimizer::TrainLoop() {
	for ( ; epoch <= maxEpoch; epoch++) {
		if (epoch % saveStep == 0){
			Save(); // 自动备份
		}
		MainTrainMethod(miniBatchSize); // 训练主过程
		Log(); //日志
	}
}

void Optimizer::LoadFromFile() {
	FILE *in = fopen(filePath.c_str(), "r");
	fscanf(in, "%d", &epoch);
	int sz = 0; fscanf(in, "%d", &sz);
#ifdef ENABLE_CUDA
	assert(sz == func -> param_cnt);
	func -> syncParamFromDeviceToHost();
	for (int i = 0; i < func -> param_cnt; i++) {
		fscanf(in, "%lf", (func -> cpu_param)[i]);
	}
	func -> syncParamFromHostToDevice();
#else
	assert(sz == (func -> param).size()); //保证相同
	for (int i = 0; i < (func -> param).size(); i++) {
		fscanf(in, "%lf", (func -> param)[i]);
	}
#endif
	fclose(in);
}

void Optimizer::Log() {
	printf("epoch: %d/%d, loss: %lf\n", epoch, maxEpoch, meanLoss);
	FOR(i, 1, 10) {
		printf("%.3f ", (*(func -> Output))(1, i).forwardBuffer[1]);
	} puts("");
}

void Optimizer::TrainFromNothing() {
	func -> Randomization(seed, randL, randR, randEps);
	epoch = 1;
	TrainLoop();
}

void Optimizer::TrainFromFile() {
	LoadFromFile();
	TrainLoop();
}

void Optimizer::SetSaveStep(int step) {
	this -> saveStep = step;
}

void Optimizer::Save() {
	FILE* out = fopen(filePath.c_str(), "w");
	fprintf(out, "%d\n", epoch);
#ifdef ENABLE_CUDA
	fprintf(out, "%d\n", func -> param_cnt);
	func -> syncParamFromDeviceToHost();
	for (int i = 0; i < func -> param_cnt; i++) {
		fprintf(out, "%lf\n", (func -> cpu_param)[i]);
	}
#else
	fprintf(out, "%d\n", (func -> param).size());
	for (int i = 0; i < (func -> param).size(); i++) {
		fprintf(out, "%lf\n", *((func -> param)[i]));
	}
#endif
	fclose(out);
}

void Optimizer::UnitlizeVector(vector<double> &data) {
	double del = 0;
	for (int i = 0; i < data.size(); i++) del += data[i] * data[i];
	del = sqrt(del);
	for (int i = 0; i < data.size(); i++) data[i] /= del;
}

void Optimizer::MainTrainMethod(int batchSize) { // 一次迭代中的计算过程 (SGD)
	vector<int> seqence;
	seqence.resize(trainData.size(), 0);
	for (int i = 0; i < trainData.size(); i++) seqence[i] = i;
	random_shuffle(seqence.begin(), seqence.end());

	meanLoss = 0;
	assert((func -> param).size() == direction.size());
	for (int i = 0; i < direction.size(); i++) direction[i] = 0;

	for (int i = 0; i < batchSize; i++) {
		func -> Update(trainData[seqence[i]], trainLabel[seqence[i]]);
		meanLoss += func -> GetLoss();
#ifdef ENABLE_CUDA
		func -> syncParamFromDeviceToHost();
		for (int j = 0; j < direction.size(); j++) {
			direction[j] += (func -> cpu_paramDel)[i];
		}
#else
		for (int j = 0; j < direction.size(); j++) {
			direction[j] += *((func->paramDel)[j]);
		}
#endif
	}

	UnitlizeVector(direction);
	meanLoss /= batchSize;

#ifdef ENABLE_CUDA
	for (int i = 0; i < direction.size(); i++) {
		direction[i] = -step * direction[i];
		(func -> cpu_param)[i] += direction[i];
	}
	func -> syncParamFromHostToDevice();
#else
	for (int i = 0; i < direction.size(); i++) {
		direction[i] = -step * direction[i];
		*((func -> param)[i]) += direction[i];
	}
#endif
}


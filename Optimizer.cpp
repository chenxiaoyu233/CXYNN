#include "Optimizer.h"
// cuda kernels
#ifdef ENABLE_CUDA
#include "cuda/kernels.h"
#endif


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
#ifdef ENABLE_CUDA
	CHECK( cudaMalloc(&gpu_direction, (func -> param_cnt) * sizeof(double)) );
#else
	direction.clear();
	direction.resize((func->param).size(), 0);
#endif
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
		fscanf(in, "%lf", func -> cpu_param + i);
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
	FOR(i, 1, 1) {
		//printf("%.3f ", (*(func -> Output))(1, i).forwardBuffer[1]);
	} //puts("");
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

#ifdef ENABLE_CUDA
void Optimizer::UnitlizeVector(double *gpu_direction, int len) {
	double del = kernel_vector_dot(gpu_direction, gpu_direction, len);
	del = sqrt(del);
	kernel_vector_mutiply(gpu_direction, 1.0f/del, len);
}
#else
void Optimizer::UnitlizeVector(vector<double> &data) {
	double del = 0;
	for (int i = 0; i < data.size(); i++) del += data[i] * data[i];
	del = sqrt(del);
	for (int i = 0; i < data.size(); i++) data[i] /= del;
}
#endif

void Optimizer::MainTrainMethod(int batchSize) { // 一次迭代中的计算过程 (SGD)
	vector<int> seqence;
	seqence.resize(trainData.size(), 0);
	for (int i = 0; i < trainData.size(); i++) seqence[i] = i;
	random_shuffle(seqence.begin(), seqence.end());

	meanLoss = 0;
#ifdef ENABLE_CUDA
	kernel_vector_set_zero(gpu_direction, func -> param_cnt);
#else
	assert((func -> param).size() == direction.size());
	for (int i = 0; i < direction.size(); i++) direction[i] = 0;
#endif

	for (int i = 0; i < batchSize; i++) {
		func -> Update(trainData[seqence[i]], trainLabel[seqence[i]]);
		meanLoss += func -> GetLoss();
#ifdef ENABLE_CUDA
		kernel_sync_param_from_device_to_host(func -> gpu_paramDel_ptr, func -> gpu_paramDel, func -> param_cnt);
		kernel_vector_add_to(gpu_direction, func -> gpu_paramDel, func -> param_cnt);
#else
		for (int j = 0; j < direction.size(); j++) {
			direction[j] += *((func->paramDel)[j]);
		}
#endif
	}

#ifdef ENABLE_CUDA
	UnitlizeVector(gpu_direction, func -> param_cnt);
#else
	UnitlizeVector(direction);
#endif
	meanLoss /= batchSize;

#ifdef ENABLE_CUDA
	kernel_vector_add_to_with_factor(func -> gpu_param, gpu_direction, func -> param_cnt, -step);
	kernel_sync_param_from_host_to_device(func -> gpu_param_ptr, func -> gpu_param, func -> param_cnt);
#else
	for (int i = 0; i < direction.size(); i++) {
		direction[i] = -step * direction[i];
		*((func -> param)[i]) += direction[i];
	}
#endif
}


#include "FuncAbstractor.h"

FuncAbstractor::FuncAbstractor(
	Layer* Input,
   	Layer* Output,
   	Estimator* estimator,
	double L2param

):  Input(Input),
   	Output(Output),
   	estimator(estimator),
	L2param(L2param) {

	param.clear();
	paramDel.clear();
	Output -> CollectParam(&(this -> param), &(this -> paramDel));
#ifdef ENABLE_CUDA
	param_cnt = param.size();
	cpu_param = new double[param_cnt];
	cpu_paramDel = new double[param_cnt];
	cudaMalloc(&gpu_param_ptr, sizeof(double**) * param_cnt);
	cudaMalloc(&gpu_paramDel_ptr, sizeof(double**) * param_cnt);
	cudaMalloc(&gpu_param, sizeof(double) * param_cnt);
	cudaMalloc(&gpu_paramDel, sizeof(double) * param_cnt);

	cpu_param_ptr = new (double*)[param_cnt];
	cpu_paramDel_ptr = new (double*)[param_cnt];
	FOR(i, 0, param_cnt-1) {
		cpu_param_ptr[i] = param[i];
		cpu_paramDel_ptr[i] = paramDel[i];
	}
	cudaMemcpy(gpu_param_ptr, cpu_param_ptr, sizeof(double*) * param_cnt, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_paramDel_ptr, cpu_paramDel_ptr, sizeof(double*) * param_cnt, cudaMemcpyHostToDevice);
	delete[] cpu_param_ptr;
	delete[] cpu_paramDel_ptr;
#endif
	loss = 0;
}

#ifdef ENABLE_CUDA
FuncAbstractor::~FuncAbstractor() {
	delete[] cpu_param;
	delete[] cpu_paramDel;
	cudaFree(gpu_param_ptr);
	cudaFree(gpu_paramDel_ptr);
	cudaFree(gpu_param);
	cudaFree(gpu_paramDel);
}

void FuncAbstractor::syncParamFromHostToDevice() {
	cudaMemcpy(gpu_param, cpu_param, sizeof(double) * param_cnt, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_paramDel, cpu_paramDel, sizeof(double) * param_cnt, cudaMemcpyHostToDevice);
	kernel_sync_param_from_host_to_device(gpu_param_ptr, gpu_param, param_cnt);
	kernel_sync_param_from_host_to_device(gpu_paramDel_ptr, gpu_paramDel, param_cnt);
}

void FuncAbstractor::syncParamFromDeviceToHost() {
	kernel_sync_param_from_device_to_host(gpu_param_ptr, gpu_param, param_cnt);
	kernel_sync_param_from_device_to_host(gpu_paramDel_ptr, gpu_paramDel, param_cnt);
	cudaMemcpy(cpu_param, gpu_param, sizeof(double) * param_cnt, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_paramDel, gpu_paramDel, sizeof(double) * param_cnt, cudaMemcpyDeviceToHost);
}
#endif

void FuncAbstractor::L2regularization() {
#ifdef ENABLE_CUDA
	syncParamFromDeviceToHost();
	double del = 0;
	for (int i = 0; i < param_cnt; i++) {
		del += cpu_param[i] * cpu_param[i];
	}
	del /= 2 * param_cnt;
	del *= L2param;
	loss += del;
	for (int i = 0; i < param_cnt; i++) {
		cpu_paramDel[i] += cpu_param[i] * L2param / param_cnt;
	}
	syncParamFromHostToDevice();
#else
	// 正则化Loss 
	double del = 0;
	for (int i = 0; i < param.size(); i++) {
		del += (*param[i]) * (*param[i]);
	}
	del /= 2 * param.size();
	del *= L2param;
	loss += del;

	// 正则化偏导数
	for (int i = 0; i < paramDel.size(); i++) {
		(*paramDel[i]) += (*param[i]) * L2param / param.size();
	}
#endif
}

void FuncAbstractor::Randomization(int seed, int l, int r) {
	srand(seed);
#ifdef ENABLE_CUDA
	syncParamFromDeviceToHost();
	for (int i = 0; i < param_cnt; i++) {
		cpu_param[i] = Tools.RandInt(l, r);
	}
	syncParamFromHostToDevice();
#else
	for (int i = 0; i < param.size(); i++) {
		*(param[i]) = Tools.RandInt(l, r);
	}
#endif
}

void FuncAbstractor::Randomization(int seed, double l, double r, double eps) {
	srand(seed);
#ifdef ENABLE_CUDA
	syncParamFromDeviceToHost();
	for (int i = 0; i < param_cnt; i++) {
		cpu_param[i] = Tools.RandDouble(l, r, eps);
	}
	syncParamFromHostToDevice();
#else
	for (int i = 0; i < param.size(); i++) {
		*(param[i]) = Tools.RandDouble(l, r, eps);
	}
#endif
}

void FuncAbstractor::ClearParamDel() {
#ifdef ENABLE_CUDA
	syncParamFromDeviceToHost();
	for (int i = 0; i < param_cnt; i++) cpu_paramDel[i] = 0;
	syncParamFromHostToDevice();
#else
	for (int i = 0; i < paramDel.size(); i++) *(paramDel[i]) = 0;
#endif
}

void FuncAbstractor::Update(Matrix<double> *dataIn, Matrix<double> *expect) {
	ClearParamDel();
	Input -> UpdateForwardBegin(dataIn); // 前向更新值
	loss = estimator -> Loss(expect); //计算loss
	L2regularization();
	estimator -> LossDel(expect); //更新输出层的backwardBuffer
	Output -> SpreadBack(); //反向传播
}

double FuncAbstractor::GetLoss() {
	return loss;
}

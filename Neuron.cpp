#include "Neuron.h"
// cuda kernels
#ifdef ENABLE_CUDA
#include "cuda/kernels.h"
#endif


Neuron::Neuron() {
	ActiveFunc = NULL;
	ActiveFuncDelta = NULL;
#ifdef ENABLE_CUDA
	input = new vector<Fiber>; input -> clear();
	output = new vector<Fiber>; output -> clear();
	gpu_input = gpu_output = cpu_input = cpu_output = NULL;
	gpu_input_count = gpu_output_count = cpu_input_count = cpu_output_count = NULL;
	gpu_input_idx = gpu_output_idx = cpu_input_idx = cpu_output_idx = NULL;
#else
	input.clear(); output.clear();
#endif
	b = NULL; bDel = NULL;
	dropOutFlag = false;
	forwardBuffer[0] = forwardBuffer[1] = forwardBuffer[2] = 0;
	backwardBuffer = 0;
}

Neuron::Neuron(
	double (*ActiveFunc) (double),
	double (*ActiveFuncDelta) (double),
	double* b,
	double* bDel
): ActiveFunc(ActiveFunc), ActiveFuncDelta(ActiveFuncDelta), b(b), bDel(bDel){  
#ifdef ENABLE_CUDA
	input = new vector<Fiber>; input -> clear();
	output = new vector<Fiber>; output -> clear();
	gpu_input = gpu_output = cpu_input = cpu_output = NULL;
	gpu_input_count = gpu_output_count = cpu_input_count = cpu_output_count = NULL;
	gpu_input_idx = gpu_output_idx = cpu_input_idx = cpu_output_idx = NULL;
#else
	input.clear(); output.clear();
#endif
	dropOutFlag = false;
	forwardBuffer[0] = forwardBuffer[1] = forwardBuffer[2] = 0;
	backwardBuffer = 0;
}

#ifdef ENABLE_CUDA
Neuron::~Neuron() {
	if(gpu_input != NULL) { CHECK( cudaFree(gpu_input) ); }
	if(gpu_output != NULL) { CHECK( cudaFree(gpu_output) ); }
	if(cpu_input != NULL) { delete[] cpu_input; }
	if(cpu_output != NULL) { delete[] cpu_output; }
	if(gpu_input_count != NULL) { CHECK( cudaFree(gpu_input_count) ); }
	if(gpu_output_count != NULL) { CHECK( cudaFree(gpu_output_count) ); }
	if(cpu_input_count != NULL) { delete cpu_input_count; }
	if(cpu_output_count != NULL) { delete cpu_output_count; }
	if(cpu_input_idx != NULL) { delete[] cpu_input_idx; }
	if(cpu_output_idx != NULL) { delete[] cpu_output_idx; }
	if(gpu_input_idx != NULL) { CHECK( cudaFree(gpu_input_idx) ); }
	if(gpu_output_idx != NULL) { CHECK( cudaFree(gpu_output_idx) ); }
}
#endif

void Neuron::Insert(
	NeuronIO type, 
	double* weight,
	double* dWeight, 
	Neuron* neighbor
) {
#ifdef ENABLE_CUDA
	if (type == INPUT) input -> push_back(Fiber(weight, dWeight, neighbor));
	if (type == OUTPUT) output -> push_back(Fiber(weight, dWeight, neighbor));
#else
	if (type == INPUT) input.push_back(Fiber(weight, dWeight, neighbor));
	if (type == OUTPUT) output.push_back(Fiber(weight, dWeight, neighbor));
#endif
}

#ifndef ENABLE_CUDA
void Neuron::SetValue(double x){
	x += *b;
	forwardBuffer[0] = x;
	forwardBuffer[1] = ActiveFunc(x);
	forwardBuffer[2] = ActiveFuncDelta(x);
}
#endif

void Neuron::SetActionFunc(
		double (*ActiveFunc) (double),
		double (*ActiveFuncDelta) (double)
) {
	this -> ActiveFunc = ActiveFunc;
	this -> ActiveFuncDelta = ActiveFuncDelta;
}

#ifndef ENABLE_CUDA
void Neuron::UpdateBuffer() {
	forwardBuffer[0] = 0;
	for (int i = 0; i < input.size(); i++) {
		forwardBuffer[0] += (input[i].neuron -> forwardBuffer[1]) * (*input[i].weight);
	}
	forwardBuffer[0] += *b;
	forwardBuffer[1] = ActiveFunc(forwardBuffer[0]);
	forwardBuffer[2] = ActiveFuncDelta(forwardBuffer[0]);
}

void Neuron::SpreadBack() {
	backwardBuffer = 0;
	for (int i = 0; i < output.size(); i++) {
		backwardBuffer += (output[i].neuron -> backwardBuffer) * (*output[i].weight) * forwardBuffer[2];
		*output[i].weightDel += (output[i].neuron -> backwardBuffer) * forwardBuffer[1];
		//同一个值可能被多次用到, 直接累加
	}
	*bDel += backwardBuffer;
}
#endif

void Neuron::Log() {
#ifdef ENABLE_CUDA
// 暂时不需要日志
#else
	printf("%.2lf %.2lf %.2lf\n", forwardBuffer[0], forwardBuffer[1], forwardBuffer[2]);
	printf("%.2lf %.2lf\n", backwardBuffer);
	for (int i = 0; i < input.size(); i++) {
		input[i].Log();
	}
	for (int i = 0; i < output.size(); i++) {
		output[i].Log();
	}
	printf("%.2lf %.2lf\n", b, bDel);
#endif
}

#ifdef ENABLE_CUDA
void Neuron::SyncFiberInfo() {
	syncFiberInfo(
		input,
		&cpu_input, &gpu_input,
		&cpu_input_count, &gpu_input_count,
		&cpu_input_idx, &gpu_input_idx
	);
	syncFiberInfo(
		output,
		&cpu_output, &gpu_output,
		&cpu_output_count, &gpu_output_count,
		&cpu_output_idx, &gpu_output_idx
	);
}

void Neuron::syncFiberInfo(
	vector<Fiber> *vec,
	Fiber **fiber, Fiber **gpu_fiber, 
	int **cpu_count, int **gpu_count,
	int **cpu_idx, int **gpu_idx
) {
	int cnt = vec -> size();
	*cpu_count = new int; 
	**cpu_count = cnt;
	*fiber = new Fiber[cnt];
	*cpu_idx = new int[cnt];

	FOR(i, 0, cnt-1) {
		(*fiber)[i] = (*vec)[i];
		(*cpu_idx)[i] = (*fiber)[i].neuron -> idx;
	}

	CHECK( cudaMalloc(gpu_count, sizeof(int)) );
	CHECK( cudaMalloc(gpu_fiber, sizeof(Fiber) * cnt) );
	CHECK( cudaMalloc(gpu_idx, sizeof(int) * cnt) );

	CHECK( cudaMemcpy(*gpu_count, *cpu_count, sizeof(int), cudaMemcpyHostToDevice) );
	CHECK( cudaMemcpy(*gpu_fiber, *fiber, sizeof(Fiber) * cnt, cudaMemcpyHostToDevice) );
	CHECK( cudaMemcpy(*gpu_idx, *cpu_idx, sizeof(int) * cnt, cudaMemcpyHostToDevice) );
}
#endif

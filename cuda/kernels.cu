#include "kernels.h"

int dev = 0;
cudaDeviceProp devProp;
/*
使用GPU device " << dev << ": " << devProp.name << std::endl;
SM的数量：" << devProp.multiProcessorCount << std::endl;
每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
每个SM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
每个SM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
*/

void kernel_init_info() {
	cudaGetDeviceProperties(&devProp, dev);
}

int block_size(){
	return devProp.maxThreadsPerBlock;
}

int grid_size(int N){
	return (N + block_size() - 1) / block_size();
}

__global__ void __rebuild_input_fiber__ (Neuron *input_field, Neuron *gpu_field, int len) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= len) return;
	for(int i = 0; i < gpu_field[idx].gpu_input_count; ++i) {
		gpu_field[idx].gpu_input[i].neuron = input_field + gpu_field[idx].gpu_input_idx[i];
	}
}

void kernel_rebuild_input_fiber(Neuron *input_field, Neuron *gpu_field, int len) {
	__rebuild_input_fiber__<<<grid_size(len), block_size()>>>(input_field, gpu_field, len);
	cudaDeviceSynchronize();
}

__global__ void __rebuild_output_fiber__ (Neuron *output_field, Neuron *gpu_field, int len) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= len) return;
	for(int i = 0; i < gpu_field[idx].gpu_output_count; ++i) {
		gpu_field[idx].gpu_output[i].neuron = output_field + gpu_field[idx].gpu_output_idx[i];
	}
}

void kernel_rebuild_output_fiber(Neuron *output_field, Neuron *gpu_field, int len) {
	__rebuild_output_fiber__<<<grid_size(len), block_size()>>>(output_field, gpu_field, len);
	cudaDeviceSynchronize();
}

__global__ void __update_forward__ (Neuron *gpu_field, int len) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= len) return;
	gpu_field[idx].forwardBuffer[0] = 0;
	for(int i = 0; i < gpu_field[idx].gpu_input_count; ++i) {
		gpu_field[idx].forwardBuffer[0] += 
			(gpu_field[idx].gpu_input[i].neuron -> forwardBuffer[1]) * (*(gpu_field[idx].gpu_input[i].weight));
	}
	gpu_field[idx].forwardBuffer[0] += *(gpu_field[idx].b);
	gpu_field[idx].forwardBuffer[1] = gpu_field[idx].ActiveFunc(gpu_field[idx].forwardBuffer[0]);
	gpu_field[idx].forwardBuffer[2] = gpu_field[idx].ActiveFuncDelta(gpu_field[idx].forwardBuffer[0]);
}

void kernel_update_forward(Neuron *gpu_field, int len) {
	__update_forward__<<<grid_size(len), block_size()>>>(gpu_field, len);
	cudaDeviceSynchronize();
}

__global__ void __spread_back__ (Neuron *gpu_field, int len) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= len) return;
	gpu_field[idx].backwardBuffer = 0;
	for(int i = 0; i < gpu_field[idx].gpu_output_count; ++i) {
		gpu_field[idx].backwardBuffer += 
			(gpu_field[idx].gpu_output[i].neuron -> backwardBuffer) * 
			(*(gpu_field[idx].gpu_output[i].weight)) *
			gpu_field[idx].forwardBuffer[2];
		*(gpu_field[idx].gpu_output[i].weightDel) += 
			(gpu_field[idx].gpu_output[i].neuron -> backwardBuffer) *
			gpu_field[idx].forwardBuffer[1];
	}
	*(gpu_field[idx].bDel) += gpu_field[idx].backwardBuffer;
}

void kernel_spread_back(Neuron *gpu_field, int len) {
	__spread_back__<<<grid_size(len), block_size()>>>(gpu_field, len);
	cudaDeviceSynchronize();
}

__global__ void __sync_BackwardBuffer_to_bDel__ (Neuron *gpu_field, int len) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= len) return;
	*(gpu_field[idx].bDel) = gpu_field[idx].backwardBuffer;
}

void kernel_sync_BackwardBuffer_to_bDel(Neuron *gpu_field, int len) {
	__sync_BackwardBuffer_to_bDel__<<<grid_size(len), block_size()>>>(gpu_field, len);
	cudaDeviceSynchronize();
}

__global__ void __sync_param_from_host_to_device__ (double **param_ptr, double *param, int cnt) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= cnt) return;
	*(param_ptr[idx]) = param[idx];
}

void kernel_sync_param_from_host_to_device(double **param_ptr, double *param, int cnt) {
	__sync_param_from_host_to_device__<<<grid_size(cnt), block_size()>>>(param_ptr, param cnt);
	cudaDeviceSynchronize();
}

__global__ void __sync_param_from_device_to_host__ (double **param_ptr, double *param, int cnt) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= cnt) return;
	param[idx] = *(param_ptr[idx]);
}

void kernel_sync_param_from_device_to_host(double **param_ptr, double *param, int cnt) {
	__sync_param_from_device_to_host__<<<grid_size(cnt), block_size()>>>(param_ptr, param, cnt);
	cudaDeviceSynchronize();
}

__global__ void __layer_set_value__ (Neuron *gpu_field, double *buffer, int len) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= len) return;
	gpu_field[idx].forwardBuffer[0] = buffer[idx] + *(gpu_field[idx].b);
	gpu_field[idx].forwardBuffer[1] = gpu_field[idx].ActiveFunc(gpu_field[idx].forwardBuffer[0]);
	gpu_field[idx].forwardBuffer[2] = gpu_field[idx].ActiveFuncDelta(gpu_field[idx].forwardBuffer[0]);
}

void kernel_layer_set_value(Neuron *gpu_field, double *buffer, int len) {
	__layer_set_value__<<<grid_size(len), block_size()>>>(gpu_field, buffer, len);
	cudaDeviceSynchronize();
}

__device__ double __Sigmoid__ (double x) {
	return 1.0f/(1.0f + exp(-x));
}

__device__ double __SigmoidDel__ (double x) {
	return __Sigmoid__ (x) * (1 - __Sigmoid__ (x));
}

__device__ double __ReLU__ (double x) {
	return x < 0 ? 0 : x;
}

__device__ double __ReLUDel__ (double x) {
	if (x <= 0) return 0;
	if (x > 0) return 1;
}

__device__ double __tanh__ (double x) {
	return (exp(2*x) - 1) / (exp(2*x) + 1);
}

__device__ double __tanhDel__ (double x) {
	return 1.0f - __tanh__(x) * __tanh__(x);
}

__device__ double __BNLL__ (double x) {
	return log(1.0f + exp(x));
}

__device__ double __BNLLDel__ (double x) {
	return exp(x) / (1.0f + exp(x));
}

__device__ double __Linear__ (double x) {
	return x;
}

__device__ double __LinearDel__ (double x) {
	return 1.0f;
}

void active_function_register() {
	cudaMemcpyFromSymbol(&kernel_Sigmoid, &__Sigmoid__, sizeof(&__Sigmoid__));
	cudaMemcpyFromSymbol(&kernel_SigmoidDel, &__SigmoidDel__, sizeof(&__SigmoidDel__));
	cudaMemcpyFromSymbol(&kernel_ReLU, &__ReLU__, sizeof(&__ReLU__));
	cudaMemcpyFromSymbol(&kernel_ReLUDel, &__ReLUDel__, sizeof(&__ReLUDel__));
	cudaMemcpyFromSymbol(&kernel_tanh, &__tanh__, sizeof(&__tanh__));
	cudaMemcpyFromSymbol(&kernel_tanhDel, &__tanhDel__, sizeof(&__tanhDel__));
	cudaMemcpyFromSymbol(&kernel_BNLL, &__BNLL__, sizeof(&__BNLL__));
	cudaMemcpyFromSymbol(&kernel_BNLLDel, &__BNLLDel__, sizeof(&__BNLLDel__));
	cudaMemcpyFromSymbol(&kernel_Linear, &__Linear__, sizeof(&__Linear__));
	cudaMemcpyFromSymbol(&kernel_LinearDel, &__LinearDel__, sizeof(&__LinearDel__));
}

void cuda_init() {
	kernel_init_info();
	active_function_register();
}

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
	CHECK( cudaGetDeviceProperties(&devProp, dev) );
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
	for(int i = 0; i < *(gpu_field[idx].gpu_input_count); ++i) {
		gpu_field[idx].gpu_input[i].neuron = input_field + gpu_field[idx].gpu_input_idx[i];
	}
}

void kernel_rebuild_input_fiber(Neuron *input_field, Neuron *gpu_field, int len) {
	__rebuild_input_fiber__<<<grid_size(len), block_size()>>>(input_field, gpu_field, len);
	CHECK_KERNEL();
	cudaDeviceSynchronize();
}

__global__ void __rebuild_output_fiber__ (Neuron *output_field, Neuron *gpu_field, int len) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= len) return;
	for(int i = 0; i < *(gpu_field[idx].gpu_output_count); ++i) {
		gpu_field[idx].gpu_output[i].neuron = output_field + gpu_field[idx].gpu_output_idx[i];
	}
}

void kernel_rebuild_output_fiber(Neuron *output_field, Neuron *gpu_field, int len) {
	__rebuild_output_fiber__<<<grid_size(len), block_size()>>>(output_field, gpu_field, len);
	CHECK_KERNEL();
	cudaDeviceSynchronize();
}

__global__ void __update_forward__ (Neuron *gpu_field, int len) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= len) return;
	gpu_field[idx].forwardBuffer[0] = 0;
	for(int i = 0; i < (*gpu_field[idx].gpu_input_count); ++i) {
		gpu_field[idx].forwardBuffer[0] += 
			(gpu_field[idx].gpu_input[i].neuron -> forwardBuffer[1]) * (*(gpu_field[idx].gpu_input[i].weight));
	}
	gpu_field[idx].forwardBuffer[0] += *(gpu_field[idx].b);
	gpu_field[idx].forwardBuffer[1] = gpu_field[idx].ActiveFunc(gpu_field[idx].forwardBuffer[0]);
	gpu_field[idx].forwardBuffer[2] = gpu_field[idx].ActiveFuncDelta(gpu_field[idx].forwardBuffer[0]);
}

void kernel_update_forward(Neuron *gpu_field, int len) {
	__update_forward__<<<grid_size(len), block_size()>>>(gpu_field, len);
	CHECK_KERNEL();
	cudaDeviceSynchronize();
}

__global__ void __spread_back__ (Neuron *gpu_field, int len) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= len) return;
	gpu_field[idx].backwardBuffer = 0;
	for(int i = 0; i < (*gpu_field[idx].gpu_output_count); ++i) {
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
	CHECK_KERNEL();
	cudaDeviceSynchronize();
}

__global__ void __sync_BackwardBuffer_to_bDel__ (Neuron *gpu_field, int len) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= len) return;
	*(gpu_field[idx].bDel) += gpu_field[idx].backwardBuffer;
}

void kernel_sync_BackwardBuffer_to_bDel(Neuron *gpu_field, int len) {
	__sync_BackwardBuffer_to_bDel__<<<grid_size(len), block_size()>>>(gpu_field, len);
	CHECK_KERNEL();
	cudaDeviceSynchronize();
}

__global__ void __sync_param_from_host_to_device__ (double **param_ptr, double *param, int cnt) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= cnt) return;
	*(param_ptr[idx]) = param[idx];
}

void kernel_sync_param_from_host_to_device(double **param_ptr, double *param, int cnt) {
	__sync_param_from_host_to_device__<<<grid_size(cnt), block_size()>>>(param_ptr, param, cnt);
	CHECK_KERNEL();
	cudaDeviceSynchronize();
}

__global__ void __sync_param_from_device_to_host__ (double **param_ptr, double *param, int cnt) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= cnt) return;
	param[idx] = *(param_ptr[idx]);
}

void kernel_sync_param_from_device_to_host(double **param_ptr, double *param, int cnt) {
	__sync_param_from_device_to_host__<<<grid_size(cnt), block_size()>>>(param_ptr, param, cnt);
	CHECK_KERNEL();
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
	CHECK_KERNEL();
	cudaDeviceSynchronize();
}

__global__ void __maxpool_push_spread_back__ (Neuron *gpu_field, int len) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= len) return;
	for(int i = 0; i < *(gpu_field[idx].gpu_input_count); ++i) {
		if(gpu_field[idx].gpu_input[i].neuron -> forwardBuffer[1] == gpu_field[idx].forwardBuffer[0]) {
			gpu_field[idx].gpu_input[i].neuron -> backwardBuffer += gpu_field[idx].backwardBuffer;
		}
	}
}

void kernel_maxpool_push_spread_back(Neuron *gpu_field, int len) {
	__maxpool_push_spread_back__<<<grid_size(len), block_size()>>>(gpu_field, len);
	CHECK_KERNEL();
	cudaDeviceSynchronize();
}

__global__ void __maxpool_update_forward__ (Neuron *gpu_field, int len) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= len) return;
	gpu_field[idx].forwardBuffer[0] = -1e10; // 标兵
	for(int i = 0; i < *(gpu_field[idx].gpu_input_count); ++i) {
		gpu_field[idx].forwardBuffer[0] = 
			max(gpu_field[idx].forwardBuffer[0], gpu_field[idx].gpu_input[i].neuron -> forwardBuffer[1]);
	}
	gpu_field[idx].forwardBuffer[1] = gpu_field[idx].forwardBuffer[0];
	gpu_field[idx].forwardBuffer[2] = 1.0f;
}

void kernel_maxpool_update_forward(Neuron *gpu_field, int len) {
	__maxpool_update_forward__<<<grid_size(len), block_size()>>>(gpu_field, len);
	CHECK_KERNEL();
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
	else return 1;
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

__device__ ACTFUNC dev_Sigmoid = __Sigmoid__;
__device__ ACTFUNC dev_SigmoidDel = __SigmoidDel__;

__device__ ACTFUNC dev_ReLU = __ReLU__;
__device__ ACTFUNC dev_ReLUDel = __ReLUDel__;

__device__ ACTFUNC dev_tanh = __tanh__;
__device__ ACTFUNC dev_tanhDel = __tanhDel__;

__device__ ACTFUNC dev_BNLL = __BNLL__;
__device__ ACTFUNC dev_BNLLDel = __BNLLDel__;

__device__ ACTFUNC dev_Linear = __Linear__;
__device__ ACTFUNC dev_LinearDel = __LinearDel__;

ACTFUNC kernel_Sigmoid;
ACTFUNC kernel_SigmoidDel;

ACTFUNC kernel_ReLU;
ACTFUNC kernel_ReLUDel;

ACTFUNC kernel_tanh;
ACTFUNC kernel_tanhDel;

ACTFUNC kernel_BNLL;
ACTFUNC kernel_BNLLDel;

ACTFUNC kernel_Linear;
ACTFUNC kernel_LinearDel;

void active_function_register() {
	CHECK( cudaMemcpyFromSymbol(&kernel_Sigmoid, dev_Sigmoid, sizeof(ACTFUNC)) );
	CHECK( cudaMemcpyFromSymbol(&kernel_SigmoidDel, dev_SigmoidDel, sizeof(ACTFUNC)) );
	CHECK( cudaMemcpyFromSymbol(&kernel_ReLU, dev_ReLU, sizeof(ACTFUNC)) );
	CHECK( cudaMemcpyFromSymbol(&kernel_ReLUDel, dev_ReLUDel, sizeof(ACTFUNC)) );
	CHECK( cudaMemcpyFromSymbol(&kernel_tanh, dev_tanh, sizeof(ACTFUNC)) );
	CHECK( cudaMemcpyFromSymbol(&kernel_tanhDel, dev_tanhDel, sizeof(ACTFUNC)) );
	CHECK( cudaMemcpyFromSymbol(&kernel_BNLL, dev_BNLL, sizeof(ACTFUNC)) );
	CHECK( cudaMemcpyFromSymbol(&kernel_BNLLDel, dev_BNLLDel, sizeof(ACTFUNC)) );
	CHECK( cudaMemcpyFromSymbol(&kernel_Linear, dev_Linear, sizeof(ACTFUNC)) );
	CHECK( cudaMemcpyFromSymbol(&kernel_LinearDel, dev_LinearDel, sizeof(ACTFUNC)) );
}

void cuda_init() {
	kernel_init_info();
	active_function_register();
}

__global__ void __debug_print_int__ (int *x) {
	printf("%d\n", *x);
}

void kernel_debug_print_int(int *x) {
	__debug_print_int__ <<<1, 1>>> (x);
	CHECK_KERNEL();
	cudaDeviceSynchronize();
}

__global__ void __debug_print_double__ (double *x) {
	printf("%lf\n", *x);
}

void kernel_debug_print_double(double *x) {
	__debug_print_double__ <<<1, 1>>> (x);
	CHECK_KERNEL();
	cudaDeviceSynchronize();
}

__global__ void __debug_print_ptr_double__ (double **x) {
	printf("%lf\n", **x);
}

void kernel_debug_print_ptr_double(double **x) {
	__debug_print_ptr_double__ <<<1, 1>>> (x);
	CHECK_KERNEL();
	cudaDeviceSynchronize();
}

__global__ void __debug_print_ptr__ (void **x) {
	printf("0x%p\n", *x);
}

void kernel_debug_print_ptr(void **x) {
	__debug_print_ptr__ <<<1, 1>>> (x);
	CHECK_KERNEL();
	cudaDeviceSynchronize();
}

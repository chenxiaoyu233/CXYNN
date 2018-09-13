#include "kernels.h"
// curand库
#include <curand.h>
#include <curand_kernel.h>

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

#define MAX_BLOCK_SIZE 512

int block_size(){
	return min(MAX_BLOCK_SIZE, devProp.maxThreadsPerBlock);
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
	//CHECK( cudaDeviceSynchronize() );
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
	//CHECK( cudaDeviceSynchronize() );
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
	//CHECK( cudaDeviceSynchronize() );
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
	//CHECK( cudaDeviceSynchronize() );
}

__global__ void __sync_BackwardBuffer_to_bDel__ (Neuron *gpu_field, int len) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= len) return;
	*(gpu_field[idx].bDel) += gpu_field[idx].backwardBuffer;
}

void kernel_sync_BackwardBuffer_to_bDel(Neuron *gpu_field, int len) {
	__sync_BackwardBuffer_to_bDel__<<<grid_size(len), block_size()>>>(gpu_field, len);
	CHECK_KERNEL();
	//CHECK( cudaDeviceSynchronize() );
}

__global__ void __sync_param_from_host_to_device__ (double **param_ptr, double *param, int cnt) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= cnt) return;
	*(param_ptr[idx]) = param[idx];
}

void kernel_sync_param_from_host_to_device(double **param_ptr, double *param, int cnt) {
	__sync_param_from_host_to_device__<<<grid_size(cnt), block_size()>>>(param_ptr, param, cnt);
	CHECK_KERNEL();
	//CHECK( cudaDeviceSynchronize() );
}

__global__ void __sync_param_from_device_to_host__ (double **param_ptr, double *param, int cnt) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= cnt) return;
	param[idx] = *(param_ptr[idx]);
}

void kernel_sync_param_from_device_to_host(double **param_ptr, double *param, int cnt) {
	__sync_param_from_device_to_host__<<<grid_size(cnt), block_size()>>>(param_ptr, param, cnt);
	CHECK_KERNEL();
	//CHECK( cudaDeviceSynchronize() );
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
	//CHECK( cudaDeviceSynchronize() );
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
	//CHECK( cudaDeviceSynchronize() );
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
	//CHECK( cudaDeviceSynchronize() );
}

// 全局自适应显存(用于向量操作，不会释放)
double *vector_gpu_temp_memory = NULL;
double *vector_cpu_temp_memory = NULL;
int vector_temp_memory_size = 0;

void kernel_check_vector_operation_memory(int size) {
	if(vector_temp_memory_size < size) {
		if(vector_gpu_temp_memory != NULL) CHECK( cudaFree(vector_gpu_temp_memory) );
		if(vector_cpu_temp_memory != NULL) delete[] vector_cpu_temp_memory;
		CHECK( cudaMalloc(&vector_gpu_temp_memory, sizeof(double) * size) );
		vector_cpu_temp_memory = new double[size];
		vector_temp_memory_size = size;
	}
}

__global__ void __vector_set_zero__ (double *vec, int len) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= len) return;
	vec[idx] = 0.0f;
}

void kernel_vector_set_zero(double *vec, int len) {
	__vector_set_zero__<<<grid_size(len), block_size()>>>(vec, len);
	CHECK_KERNEL();
	//CHECK( cudaDeviceSynchronize() );
}

__global__ void __vector_block_dot__ (double *vec_a, double *vec_b, double *vec_ret, int len) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	__shared__ double tmp[MAX_BLOCK_SIZE];
	if(idx < len) tmp[threadIdx.x] = vec_a[idx] * vec_b[idx];
	else tmp[threadIdx.x] = 0;
	__syncthreads();

	// 树形累加 O(log n)
	// 必须保证 blockDim.x 是 2^n 的形式, 不然这样写有问题
	// 不然, 奇数序列最中间的元素会被两个线程同时操作, 数据不能同步
	for(int step = (blockDim.x >> 1); step > 0; step >>= 1) {
		if(threadIdx.x < step)
			tmp[threadIdx.x] += tmp[threadIdx.x + step];
		__syncthreads();
	}

	if(threadIdx.x == 0) vec_ret[blockIdx.x] = tmp[0];
}

double kernel_vector_dot(double *vec_a, double *vec_b, int len) { // need test and debug
	kernel_check_vector_operation_memory(grid_size(len));
	__vector_block_dot__<<<grid_size(len), block_size()>>>(vec_a, vec_b, vector_gpu_temp_memory, len);
	CHECK_KERNEL();
	//CHECK( cudaDeviceSynchronize() );
	CHECK( cudaMemcpy(vector_cpu_temp_memory, vector_gpu_temp_memory, grid_size(len) * sizeof(double), cudaMemcpyDeviceToHost) );
	double ret = 0;
	int __grid_size__ = grid_size(len);
	for(int i = 0; i < __grid_size__; ++i) 
		ret += vector_cpu_temp_memory[i];
	return ret;
}

__global__ void __vector_add_to_with_factor__ (double *dst, double *src, int len, double factor) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= len) return;
	dst[idx] += src[idx] * factor;
}

void kernel_vector_add_to_with_factor(double *dst, double *src, int len, double factor) {
	__vector_add_to_with_factor__<<<grid_size(len), block_size()>>>(dst, src, len, factor);
	CHECK_KERNEL();
	//CHECK( cudaDeviceSynchronize() );
}

__global__ void __vector_add_to__ (double *dst, double *src, int len) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= len) return;
	dst[idx] += src[idx];
}

void kernel_vector_add_to(double *dst, double *src, int len) {
	__vector_add_to__<<<grid_size(len), block_size()>>>(dst, src, len);
	CHECK_KERNEL();
	//CHECK( cudaDeviceSynchronize() );
}

__global__ void __vector_mutiply__ (double *vec, double factor, int len) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= len) return;
	vec[idx] *= factor;
}

void kernel_vector_mutiply(double *vec, double factor, int len) {
	__vector_mutiply__<<<grid_size(len), block_size()>>>(vec, factor, len);
	CHECK_KERNEL();
	//CHECK( cudaDeviceSynchronize() );
}

__global__ void __vector_double_ptr_set_value__ (double **dp, double value, int st, int ed) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x + st;
	if(idx >= ed) return;
	*(dp[idx]) = value;
}

void kernel_vector_double_ptr_set_value(double **dp, double value, int st, int ed) {
	__vector_double_ptr_set_value__<<<grid_size(ed-st), block_size()>>>(dp, value, st, ed);
	CHECK_KERNEL();
	//CHECK( cudaDeviceSynchronize() );
}

curandState_t* rand_kernel = NULL;
int rand_kernel_count = 0;

__global__ void __setup_rand_kernel__ (curandState_t *rand_kernel, int len) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= len) return;
	curand_init (2333, idx, 0, rand_kernel + idx);
}

void kernel_setup_rand_kernel(int len) {
	if(rand_kernel_count < len) {
		if(rand_kernel != NULL) CHECK( cudaFree(rand_kernel) );
		CHECK( cudaMalloc(&rand_kernel, sizeof(curandState_t) * len) );
		rand_kernel_count = len;
		__setup_rand_kernel__ <<<grid_size(len), block_size()>>>(rand_kernel, len);
		CHECK_KERNEL();
		//CHECK( cudaDeviceSynchronize() );
	}
}

__global__ void __vector_double_ptr_rand_zero_one__ (double **dp, double rate, curandState_t *state, int st, int ed) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idx_s = idx + st;
	if(idx_s >= ed) return;
	double tmp = curand_uniform(state + idx);
	if(tmp <= rate) *(dp[idx_s]) = 1;
	else *(dp[idx_s]) = 0;
}

void kernel_vector_double_ptr_rand_zero_one(double **dp, double rate, int st, int ed) {
	kernel_setup_rand_kernel(ed-st);
	__vector_double_ptr_rand_zero_one__<<<grid_size(ed-st), block_size()>>>(dp, rate, rand_kernel, st, ed);
	CHECK_KERNEL();
	//CHECK( cudaDeviceSynchronize() );
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

void cuda_uninit() {
	if(vector_gpu_temp_memory != NULL) CHECK( cudaFree(vector_gpu_temp_memory) );
	if(vector_cpu_temp_memory != NULL) delete[] vector_gpu_temp_memory;
	if(rand_kernel != NULL) CHECK( cudaFree(rand_kernel) );
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

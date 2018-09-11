#ifndef __KERNELS_H__
#define __KERNELS_H__

#include "../Neuron.h"

void kernel_init_info();
void kernel_rebuild_input_fiber(
	Neuron *input_field, 
	Neuron *gpu_field, 
	int len
);
void kernel_rebuild_output_fiber(
	Neuron *output_field,
   	Neuron *gpu_field,
   	int len
);
void kernel_update_forward(Neuron *gpu_field, int len);
void kernel_spread_back(Neuron *gpu_field, int len);
void kernel_sync_BackwardBuffer_to_bDel(Neuron *gpu_field, int len);
void kernel_sync_param_from_host_to_device(double **param_ptr, double *param, int cnt);
void kernel_sync_param_from_device_to_host(double **param_ptr, double *param, int cnt);
void kernel_layer_set_value(Neuron *gpu_field, double *buffer, int len);
void kernel_maxpool_push_spread_back(Neuron *gpu_field, int len);
void kernel_maxpool_update_forward(Neuron *gpu_field, int len);
void kernel_vector_set_zero(double *vec, int len);
double kernel_vector_dot(double *vec_a, double *vec_b, int len);
void kernel_vector_add_to_with_factor(double *gpu_param, double *gpu_paramDel, int len, double factor);
void kernel_vector_add_to(double *dst, double *src, int len);
void kernel_vector_mutiply(double *vec, double factor, int len);
void kernel_vector_double_ptr_set_value(double **dp, double value, int st, int ed);
void kernel_setup_rand_kernel(int len);
void kernel_vector_double_ptr_rand_zero_one(double **dp, double rate, int st, int ed);

typedef double (*ACTFUNC) (double);

extern ACTFUNC kernel_Sigmoid;
extern ACTFUNC kernel_SigmoidDel;

extern ACTFUNC kernel_ReLU;
extern ACTFUNC kernel_ReLUDel;

extern ACTFUNC kernel_tanh;
extern ACTFUNC kernel_tanhDel;

extern ACTFUNC kernel_BNLL;
extern ACTFUNC kernel_BNLLDel;

extern ACTFUNC kernel_Linear;
extern ACTFUNC kernel_LinearDel;

void active_function_register();

void cuda_init(); // cuda初始化
void cuda_uninit(); // cuda反初始化

// tools for debug
#include <cstdio>

void kernel_debug_print_int(int *x);
void kernel_debug_print_double(double *x);
void kernel_debug_print_ptr_double(double **x);
void kernel_debug_print_ptr(void **x);
#endif

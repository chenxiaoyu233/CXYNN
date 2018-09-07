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

extern double (*kernel_Sigmoid) (double x);
extern double (*kernel_SigmoidDel) (double x);

extern double (*kernel_ReLU) (double x);
extern double (*kernel_ReLUDel) (double x);

extern double (*kernel_tanh) (double x);
extern double (*kernel_tanhDel) (double x);

extern double (*kernel_BNLL) (double x);
extern double (*kernel_BNLLDel) (double x);

extern double (*kernel_Linear) (double x);
extern double (*kernel_LinearDel) (double x);

void active_function_register();

void cuda_init(); // cuda初始化

#endif

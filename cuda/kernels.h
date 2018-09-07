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

#endif

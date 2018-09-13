#ifndef __DROPOUT_LAYER_H__
#define __DROPOUT_LAYER_H__

#include "Common.h"
#include "Layer.h"

class DropoutLayer: public Layer {
	private:
	double prob; // 这个参数必须保证训练和使用时是一致的
	bool isTrain;
#ifdef ENABLE_CUDA
	double **gpu_param_ptr;
	double **cpu_param_ptr;
	int gpu_param_cnt;
#endif
	virtual void connectLayer(Layer *Input);
	void setBToZero();

	public:
	DropoutLayer(int channel, int row, int col, double prob, bool isTrain);
	~DropoutLayer();

	// 重载这个函数， 保证这一层的参数不会被收集上去
	virtual void CollectParam(vector<double*> *param, vector<double*> *paramDel);

	// 刷新边权
	void SetEdgeVal();
};

#endif

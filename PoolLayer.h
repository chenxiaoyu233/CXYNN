#ifndef __POOL_LAYER_H__
#define __POOL_LAYER_H__

#include "Common.h"
#include "ConvLayer.h"

class PoolLayer: public ConvLayer {
	protected:
	//各种Pool层的连接方式和Conv是相同的
	virtual void insertConv(Neuron *a, Neuron *b, int c, int mx, int my);
	virtual void allocateParamMem(int InChannel);
	virtual void freeCoreParamMem();
#ifdef ENABLE_CUDA
	double *fkWab, *fkdWab;
	double *fkb, *fkbDel;
#else
	double fkWab, fkdWab; //虚假的边权, 不需要被更新 (fake => fk)
	double fkb, fkbDel; // 虚假的偏置, 和偏置的偏导数
#endif

	public:
	PoolLayer(
		int channel,
		int row, int col,
		int coreRow = 2, int coreCol = 2,
		int stepRow = 2, int stepCol = 2,
		int padRow = 0, int padCol = 0
	);
	~PoolLayer();
};

#endif

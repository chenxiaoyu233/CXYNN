#ifndef __CONV_LAYER_H__
#define __CONV_LAYER_H__

#include "Common.h"
#include "Layer.h"

// 卷积层实现, 如果连接方式核卷积相似, 可以考虑从这个类型继承
class ConvLayer: public Layer {
	protected:
	//超参数
	int coreRow, coreCol; //卷积核的长和宽
	int stepRow, stepCol; //步幅的长和宽
	int padRow, padCol; // 补0的数量

	Neuron zeroSource; // 永远输出0 的虚拟神经元
	Matrix<double*> *Wab, *dWab; // 零时参数存储

	//申请卷积核空间+参数收集
	virtual void allocateCoreParamMem(int InChannel);
	//释放卷积核空间
	virtual void freeCoreParamMem();
	
	//c, mx, my是Wab,dWab中的下标, 表示需要用哪个存储空间
	virtual void insertConv(Neuron *a, Neuron *b, int c, int mx, int my);
	virtual void connectLayer(Layer* Input);

	public:
	ConvLayer(
		int channel, 
		int row, int col,
		int coreRow = 3, int coreCol = 3, //常见的规模
		int stepRow = 1, int stepCol = 1,
		int padRow = 0, int padCol = 0 
	);

};

#endif

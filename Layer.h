#ifndef __LAYER_H__
#define __LAYER_H__

#include "Common.h"
#include "Matrix.h"
#include "Neuron.h"


class Layer: public Matrix<Neuron> {
	protected:
	vector<double*> paramPool;
	vector<double*> paramDeltaPool;

	void Insert(Neuron *a, Neuron *b);

	//这个版本的接口不处理参数申请和收集工作, 用于特殊连接方式
	void Insert(Neuron *a, Neuron *b, double *Wab, double *dWab);
	void layerInit();

	// 继承出去, 用于实现各自的不同的连接方式
	virtual void connectLayer(Layer* Input) = 0;
	virtual void updateForward();
	virtual void spreadBack();

	public: 
	Layer* Input, *Output;

	Layer(int row, int col);
	Layer(int channel, int rol, int col);
	~Layer();

	void InputLayer(Layer* Input);

	virtual void SetActionFunc(double (*ActiveFunc) (double),
			           double (*ActiveFuncDelta) (double));

	virtual void UpdateForward();
	virtual void SpreadBack();

	//这个函数用于单独处理输入层
	void UpdateForwardBegin();
	void UpdateForwardBegin(Matrix<double> *other);

	//这个函数用于收集所有的参数
	void CollectParam(vector<double*> *param, vector<double*> *paramDel);
};

#endif

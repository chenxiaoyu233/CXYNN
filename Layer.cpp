#include "Layer.h"

void Layer::layerInit() {
	paramPool.clear();
	paramDeltaPool.clear();
	Input = NULL;
	Output = NULL;
	//申请空间 + 参数收集
	FOR(cur, 1, channel){ SetAt(cur); //换面
		FOR(i, 1, row) FOR(j, 1, col) {
			(*this)(i, j).b = new double;
			(*this)(i, j).bDel = new double;
			paramPool.push_back((*this)(i, j).b); //收集参数
			paramDeltaPool.push_back((*this)(i, j).bDel);
		}
	}
}

Layer::Layer(int row, int col): Matrix(row, col) { layerInit(); }

Layer::Layer(int channel, int row, int col)
	:Matrix(channel, row, col) { layerInit(); }

Layer::~Layer() {
	for(int i = 0; i < paramPool.size(); i++) {
		delete paramPool[i];
	}
	for(int i = 0; i < paramDeltaPool.size(); i++) {
		delete paramDeltaPool[i];
	}
}

void Layer::Insert(Neuron *a, Neuron *b) { // a -> b
	double *Wab, *dWab;
	Wab = new double; dWab = new double;
	a -> Insert(OUTPUT, Wab, dWab, b);
	b -> Insert(INPUT, Wab, dWab, a); // 这里的两个w没有实际意义.
	paramPool.push_back(Wab); //收集参数
	paramDeltaPool.push_back(dWab);
}

void Layer::Insert(Neuron *a, Neuron *b, double *Wab, double *dWab) {
	a -> Insert(OUTPUT, Wab, dWab, b);
	b -> Insert(INPUT, Wab, dWab, a);
}

void Layer::InputLayer(Layer* Input) {
	this -> Input = Input;
	Input -> Output = this;
	this -> connectLayer(Input); // 防止虚函数踩坑
}

void Layer::SetActionFunc(
	double (*ActiveFunc) (double),
	double (*ActiveFuncDelta) (double)
) {
	FOR(c, 1, channel) { this -> SetAt(c);
		FOR(x, 1, row) FOR(y, 1, col) {
			(*this)(x, y).SetActionFunc(ActiveFunc, ActiveFuncDelta);
		}
	}
}

void Layer::UpdateForwardBegin() {
	FOR(c, 1, channel) { this -> SetAt(c);
		FOR(x, 1, row) FOR(y, 1, col) {
			(*this)(x, y).forwardBuffer[1] = 0;
		}
	}
	if (Output != NULL) {
		Output -> UpdateForward();
	}
}

void Layer::UpdateForwardBegin(Matrix<double> *other) {
	assert(row == (other->size()).first); //保证规模相同
	assert(col == (other->size()).second);
	assert(channel == other->Channel());

	FOR(c, 1, channel) { this -> SetAt(c);
		FOR(x, 1, row) FOR(y, 1, col) {
			(*this)(x, y).SetValue((*other)(x, y));
		}
	}
	if (Output != NULL) {
		Output -> UpdateForward();
	}
}

void Layer::updateForward() {
	FOR(c, 1, channel) { this -> SetAt(c);
		FOR(x, 1, row) FOR(y, 1, col) {
			(*this)(x, y).UpdateBuffer();
		}
	}
}

void Layer::spreadBack() {
	FOR(c, 1, channel) { this -> SetAt(c);
		FOR(x, 1, row) FOR(y, 1, col) {
			(*this)(x, y).SpreadBack();
		}
	}
}

void Layer::UpdateForward() {
	this -> updateForward(); //内部
	if (Output != NULL) {
		Output -> UpdateForward();
	}
}

void Layer::SpreadBack() {
	if (Output != NULL) {
		this -> spreadBack(); //内部
	}
	if (Input != NULL) {
		Input -> SpreadBack();
	}
}

void Layer::CollectParam(vector<double*> *param, vector<double*> *paramDel) {
	assert(paramPool.size() == paramDeltaPool.size());
	for (int i = 0; i < paramPool.size(); i++) {
		param -> push_back(paramPool[i]);
		paramDel -> push_back(paramDeltaPool[i]);
	}
	if (Input != NULL) {
		Input -> CollectParam(param, paramDel);
	}
}

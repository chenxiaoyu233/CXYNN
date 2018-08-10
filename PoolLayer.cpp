#include "PoolLayer.h"

PoolLayer::PoolLayer(
	int channel,
	int row, int col,
	int coreRow, int coreCol,
	int stepRow, int stepCol,
	int padRow, int padCol

): ConvLayer(
	channel,
	row, col,
	coreRow, coreCol,
	stepRow, stepCol,
	padRow, padCol
) {
	Wab = dWab = 0; //初始化假边权
	// pool不需要产生任何的参数, 这里将所有参数清理掉
	for (int i = 0; i < paramPool.size(); i++) {
		delete paramPool[i];
		delete paramDeltaPool[i];
	}
	paramPool.clear();
	paramDeltaPool.clear();

	// 因为Neuron中更新值的方式是不好改动的, 所有这里就只有
	// 将就Neuron中的更新方式, 将b, bDel给留出来(不然会访问无效内存)
	FOR(c, 1, channel) { this -> SetAt(c);
		FOR(x, 1, row) FOR(y, 1, col) {
			(*this)(x, y).b = &fkb;
			(*this)(x, y).bDel = &fkbDel;
		}
	}
}

void PoolLayer::allocateParamMem(int InChannel) { } 
void PoolLayer::freeCoreParamMem() { }

void PoolLayer::insertConv(Neuron *a, Neuron *b, int c, int mx, int my) {
	if(c == channel) this -> Insert(a, b, &fkWab, &fkdWab); 
	// 这个方法虽然很蠢, 但是却可以避免写一次和ConvLayer十分雷同的连接
}

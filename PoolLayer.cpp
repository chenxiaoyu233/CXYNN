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
#ifdef ENABLE_CUDA
	CHECK( cudaMalloc(&fkWab, sizeof(double)) );
	CHECK( cudaMalloc(&fkdWab, sizeof(double)) );
	CHECK( cudaMalloc(&fkb, sizeof(double)) );
	CHECK( cudaMalloc(&fkbDel, sizeof(double)) );
#endif
	// pool不需要产生任何的参数, 这里将所有参数清理掉
	for (int i = 0; i < paramPool.size(); i++) {
#ifdef ENABLE_CUDA
		CHECK( cudaFree(paramPool[i]) );
		CHECK( cudaFree(paramDeltaPool[i]) );
#else
		delete paramPool[i];
		delete paramDeltaPool[i];
#endif
	}
	paramPool.clear();
	paramDeltaPool.clear();

	// 因为Neuron中更新值的方式是不好改动的, 所有这里就只有
	// 将就Neuron中的更新方式, 将b, bDel给留出来(不然会访问无效内存)
	FOR(c, 1, channel) { this -> SetAt(c);
		FOR(x, 1, row) FOR(y, 1, col) {
#ifdef ENABLE_CUDA
			(*this)(x, y).b = fkb;
			(*this)(x, y).bDel = fkbDel;
#else
			(*this)(x, y).b = &fkb;
			(*this)(x, y).bDel = &fkbDel;
#endif
		}
	}
}

PoolLayer::~PoolLayer() {
#ifdef ENABLE_CUDA
	CHECK( cudaFree(fkWab) );
	CHECK( cudaFree(fkdWab) );
	CHECK( cudaFree(fkb) );
	CHECK( cudaFree(fkbDel) );
#endif
}

void PoolLayer::allocateParamMem(int InChannel) { } 
void PoolLayer::freeCoreParamMem() { }

void PoolLayer::insertConv(Neuron *a, Neuron *b, int c, int mx, int my) {
#ifdef ENABLE_CUDA
	if(c == channel) this -> Insert(a, b, fkWab, fkdWab);
#else
	if(c == channel) this -> Insert(a, b, &fkWab, &fkdWab); 
#endif
	// 这个方法虽然很蠢, 但是却可以避免写一次和ConvLayer十分雷同的连接
}

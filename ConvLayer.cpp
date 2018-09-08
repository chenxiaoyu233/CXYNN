#include "ConvLayer.h"

ConvLayer::ConvLayer(
	int channel, 
	int row, int col,
	int coreRow, int coreCol, 
	int stepRow, int stepCol,
	int padRow, int padCol

):  Layer(channel, row, col),
    coreRow(coreRow), coreCol(coreCol),
    stepRow(stepRow), stepCol(stepCol),
    padRow(padRow), padCol(padCol) {

	zeroSource.forwardBuffer[0] = 0;
	zeroSource.forwardBuffer[1] = 0; 
	zeroSource.forwardBuffer[2] = 0;
	Wab = dWab = NULL;
}

void ConvLayer::allocateCoreParamMem(int InChannel) {
	if (Wab == NULL) Wab = new Matrix<double*>(InChannel, coreRow, coreCol);
	if (dWab == NULL) dWab = new Matrix<double*>(InChannel, coreRow, coreCol);
	FOR(c, 1, InChannel) FOR(x, 1, coreRow) FOR(y, 1, coreCol) {
#ifdef ENABLE_CUDA
		CHECK( cudaMalloc(&(*Wab)(c, x, y), sizeof(double)) );
		CHECK( cudaMalloc(&(*dWab)(c, x, y), sizeof(double)) );
#else
		(*Wab)(c, x, y) = new double; 
		(*dWab)(c, x, y) = new double;
#endif
		paramPool.push_back((*Wab)(c, x, y));
		paramDeltaPool.push_back((*dWab)(c, x, y));
	}
}

void ConvLayer::freeCoreParamMem() {
	delete Wab; delete dWab;
	Wab = dWab = NULL;
}

void ConvLayer::insertConv(Neuron *a, Neuron *b, int c, int mx, int my) {
	Insert(a, b, (*Wab)(c, mx, my), (*dWab)(c, mx, my));
}

// 以后来修改成奇偶通用的
void ConvLayer::connectLayer(Layer* Input) {
	pair<int, int> sz = Input -> size();
	int InChannel = Input -> Channel();
	int rowD = coreRow - 1, colD = coreCol - 1;
	int rowL = 1 - padRow, rowR = sz.first + padRow - rowD;
	int colL = 1 - padCol, colR = sz.second + padCol - colD;

	assert((rowR - rowL) % stepRow == 0); 
	assert((colR - colL) % stepCol == 0);
	assert((rowR - rowL) / stepRow + 1 == row);
	assert((colR - colL) / stepCol + 1 == col);

	FOR(cc, 1, channel){ this -> SetAt(cc); //cc: current channel
		//申请空间+参数收集
		this -> allocateCoreParamMem(InChannel);

		//连接
		FORS(x, rowL, rowR, stepRow) FORS(y, colL, colR, stepCol) {
			int cx = (x - rowL) / stepRow + 1;
			int cy = (y - colL) / stepCol + 1;
			FOR(c, 1, InChannel) FOR(dx, 0, rowD) FOR(dy, 0, colD) {
				int tx = x + dx, ty = y + dy;
				int mx = dx + 1, my = dy + 1;
				if (tx < 1 || tx > sz.first || ty < 1 || ty > sz.second) {
					this -> insertConv(&zeroSource, &(*this)(cx, cy), c, mx, my); // 方便继承之后重新实现
				} else {
					this -> insertConv(&(*Input)(c, tx, ty), &(*this)(cx, cy), c, mx, my);
				}
			}
		}
	}

	// 处理之前零时申请的 Wab, dWab 指向的空间
	this -> freeCoreParamMem();
}


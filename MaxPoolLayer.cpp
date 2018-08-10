#include "MaxPoolLayer.h"

MaxPoolLayer::MaxPoolLayer(
	int channel,
	int row, int col,
	int coreRow, int coreCol,
	int stepRow, int stepCol,
	int padRow, int padCol

): PoolLayer (
	channel,
	row, col,
	coreRow, coreCol,
	stepRow, stepCol,
	padRow, padCol

) { }

void MaxPoolLayer::pushSpreadBack() { //内部
	FOR(c, 1, channel) { this -> SetAt(c);
		FOR(x, 1, row) FOR(y, 1, col) {
			Neuron &cur = (*this)(x, y);
			for (int i = 0; i < cur.input.size(); i++) {
				//可能会有精度问题, 到时候注意一下
				if(cur.input[i].neuron -> forwardBuffer[1] == cur.forwardBuffer[0]) { 
					cur.input[i].neuron -> backwardBuffer += cur.backwardBuffer; //累加
				}
			}
		}
	}
}

void MaxPoolLayer::SpreadBack() { //外部
	if (Output != NULL) {
		this -> spreadBack(); //承上(Layer.cpp中实现的通用方法)
		this -> pushSpreadBack(); //启下
	}
	if (Input != NULL && Input -> Input != NULL) {
		Input -> Input -> SpreadBack(); //跳过下一层的更新, 因为下一层的值是直接由这一层给下去的
	}
}

void MaxPoolLayer::updateForward() {
	FOR(c, 1, channel){ this -> SetAt(c);
		FOR(x, 1, row) FOR(y, 1, col) {
			Neuron &cur = (*this)(x, y);
			cur.forwardBuffer[0] = -1e10; //标兵
			for (int i = 0; i < cur.input.size(); i++) {
				cur.forwardBuffer[0] = max(cur.forwardBuffer[0], cur.input[i].neuron -> forwardBuffer[1]);
			}
			cur.forwardBuffer[1] = cur.forwardBuffer[0];
			cur.forwardBuffer[2] = 1;
		}
	}
}

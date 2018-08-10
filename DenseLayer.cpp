#include "DenseLayer.h"

DenseLayer::DenseLayer(int row, int col):Layer(row, col) { }

void DenseLayer::connectLayer(Layer* Input) {
	this -> Input = Input;
	Input -> Output = this;
	pair<int, int> sz = Input -> size();
	int inChannel = Input -> Channel();
	FOR(a, 1, inChannel) FOR(b, 1, channel) {
		Input -> SetAt(a); SetAt(b); //定位分页
		FOR(x, 1, row) FOR(y, 1, col)
		FOR(tx, 1, sz.first) FOR(ty, 1, sz.second) {
			Insert(&(*Input)(tx, ty), &(*this)(x, y));
		}
	}
}

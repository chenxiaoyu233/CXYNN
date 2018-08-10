#ifndef __MATRIX_H__
#define __MATRIX_H__

#include "Common.h"


//矩阵的下标从1开始
template <class ValueType>
class Matrix {
	protected:
	ValueType ***field;
	int channel; //通道数
	int at; //当前通道
	int row, col; //行数, 列数

	void mallocMemory(); 
	void freeMemory();


	public:
	Matrix(); //暂时不分配任何空间
	Matrix(int row, int col);
	Matrix(int channel, int row, int col);
	~Matrix();

	ValueType& operator () (int c, int a, int b); //channel, row, col
	ValueType& operator () (int a, int b);
	ValueType& operator () (int a); //维度为1的时候可以使用.
	void SetAt(int x);
	pair<int, int> size();
	int Channel();
	void Log();
};

template <class ValueType>
void Matrix<ValueType>::freeMemory() {
	for (int i = 0; i < channel; i++) {
		for (int j = 0; j < row; j++) {
			delete[] field[i][j];
		}
		delete[] field[i];
	}
	delete[] field;
}

template <class ValueType>
void Matrix<ValueType>::mallocMemory() {
	field = NULL;
	field = new ValueType** [channel];
	for (int i = 0; i < channel; i++) {
		field[i] = new ValueType* [row];
		for (int j = 0; j < row; j++) {
			field[i][j] = new ValueType [col];
		}
	}
}

template <class ValueType>
Matrix<ValueType>::Matrix(int channel, int row, int col):channel(channel), row(row), col(col) {
	assert(channel > 0); assert(row > 0); assert(col > 0);
	mallocMemory();
	at = 0; //初始化
}

template <class ValueType>
Matrix<ValueType>::Matrix(int row, int col):row(row), col(col) {
	channel = 1; at = 0; //向下兼容
	assert(row > 0); assert(col > 0);
	mallocMemory();
}

template <class ValueType>
Matrix<ValueType>::Matrix() {
	field = NULL;
	channel = at = row = col = 0;
}

template <class ValueType>
Matrix<ValueType>::~Matrix() {
	freeMemory();
}

template <class ValueType>
ValueType& Matrix<ValueType>::operator () (int c, int a, int b) {
	assert(1 <= c); assert(c <= channel);
	assert(1 <= a); assert(a <= row);
	assert(1 <= b); assert(b <= col);
	return field[c-1][a-1][b-1];
}

template <class ValueType>
ValueType& Matrix<ValueType>::operator () (int a, int b) {
	assert(a <= row); assert(b <= col);
	assert(1 <= a); assert(1 <= b);
	return field[at][a-1][b-1];
}

template <class ValueType>
ValueType& Matrix<ValueType>::operator () (int a) {
	assert(row == 1 || col == 1);
	assert(1 <= a); assert(a <= max(row, col)); // @可能存在问题, 没仔细想
	if (row == 1) return field[at][0][a-1]; 
	if (col == 1) return field[at][a-1][0];
}

template <class ValueType>
void Matrix<ValueType>::SetAt(int x){ 
	assert(1 <= x); assert(x <= channel);
	at = x-1; 
}

template <class ValueType>
pair<int, int> Matrix<ValueType>::size() { return make_pair(row, col); }

template <class ValueType>
int Matrix<ValueType>::Channel() { return channel; }

template <class ValueType>
void Matrix<ValueType>::Log() {
	FOR(x, 0, row-1) {
		FOR(y, 0, col-1) printf("%.2f ", field[at][x][y]);
		cerr << "\n";
	}
}

#endif

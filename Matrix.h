#ifndef __MATRIX_H__
#define __MATRIX_H__

#include "Common.h"

// 数组定位(从 0 开始编号)
#ifdef ENABLE_CUDA
#define LOCATE(f, a, b, c) f[ (a) * row * col + (b) * col + (c) ]
#define AT(a, b, c) (a) * row * col + (b) * col + (c)
#endif

//矩阵的下标从1开始
template <class ValueType>
class Matrix {
#ifdef ENABLE_CUDA
	public:
#else
	protected:
#endif

#ifdef ENABLE_CUDA
	ValueType *field; // 内存空间
	ValueType *gpu_field; // 显存空间
#else
	ValueType ***field;
#endif
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

#ifdef ENABLE_CUDA
	void mallocGpuMemory(); // 创建出来之后会在析构函数中自动销毁
	void syncMemFromHostToDevice(); // 将内存中的数据同步到显存
	void syncMemFromDeviceToHost(); // 将显存中的数据同步到内存
#endif

	void SetAt(int x);
	pair<int, int> size();
	int Channel();
	void Log();
};

template <class ValueType>
void Matrix<ValueType>::freeMemory() {
#ifdef ENABLE_CUDA
	if(gpu_field != NULL) cudaFree(gpu_field);
	delete[] field;
#else
	for (int i = 0; i < channel; i++) {
		for (int j = 0; j < row; j++) {
			delete[] field[i][j];
		}
		delete[] field[i];
	}
	delete[] field;
#endif
}

template <class ValueType>
void Matrix<ValueType>::mallocMemory() {
#ifdef ENABLE_CUDA
	gpu_field = NULL;
	field = new ValueType[channel * row * col]; 
#else
	field = NULL;
	field = new ValueType** [channel];
	for (int i = 0; i < channel; i++) {
		field[i] = new ValueType* [row];
		for (int j = 0; j < row; j++) {
			field[i][j] = new ValueType [col];
		}
	}
#endif
}

#ifdef ENABLE_CUDA
template <class ValueType>
void Matrix<ValueType>::mallocGpuMemory() {
	CHECK( cudaMalloc(&gpu_field, sizeof(ValueType) * channel * row * col) ); 
}

template <class ValueType>
void Matrix<ValueType>::syncMemFromHostToDevice() {
	CHECK( cudaMemcpy(gpu_field, field, sizeof(ValueType) * channel * row * col, cudaMemcpyHostToDevice) );
}

template <class ValueType>
void Matrix<ValueType>::syncMemFromDeviceToHost() {
	CHECK( cudaMemcpy(field, gpu_field, sizeof(ValueType) * channel * row * col, cudaMemcpyDeviceToHost) );
}
#endif

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
#ifdef ENABLE_CUDA
	gpu_field = NULL;
#endif
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
#ifdef ENABLE_CUDA
	assert(AT(c-1, a-1, b-1) <= channel * row * col);
	return LOCATE(field, c-1, a-1, b-1);
#else
	return field[c-1][a-1][b-1];
#endif
}

template <class ValueType>
ValueType& Matrix<ValueType>::operator () (int a, int b) {
	assert(a <= row); assert(b <= col);
	assert(1 <= a); assert(1 <= b);
	//printf("cur: %d, tot: %d, a: %d, b: %d\n", AT(at, a-1, b-1), channel * row * col, a, b);
#ifdef ENABLE_CUDA
	assert(AT(at, a-1, b-1) <= channel * row * col);
	return LOCATE(field, at, a-1, b-1);
#else
	return field[at][a-1][b-1];
#endif
}

template <class ValueType>
ValueType& Matrix<ValueType>::operator () (int a) {
	assert(row == 1 || col == 1);
	assert(1 <= a); assert(a <= max(row, col)); // @可能存在问题, 没仔细想
#ifdef ENABLE_CUDA
	if (row == 1) {
		assert(AT(at, 0, a-1) <= channel * row * col);
		return LOCATE(field, at, 0, a-1); 
	}
	if (col == 1) {
		assert(AT(at, a-1, 0) <= channel * row * col);
		return LOCATE(field, at, a-1, 0);
	}
#else
	if (row == 1) return field[at][0][a-1]; 
	if (col == 1) return field[at][a-1][0];
#endif
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

#ifdef ENABLE_CUDA
template <class ValueType>
void Matrix<ValueType>::Log() { }
#else
template <class ValueType>
void Matrix<ValueType>::Log() {
	FOR(x, 0, row-1) {
		FOR(y, 0, col-1) printf("%.2f ", field[at][x][y]);
		cerr << "\n";
	}
}
#endif

#endif

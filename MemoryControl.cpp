#include "MemoryControl.h"

// 取消宏定义, 防止递归
#undef cudaMalloc
#undef cudaFree
#undef cudaMemcpy

// 常用常量
const size_t MemoryControl::__20M__ = (size_t)20 * 1024 * 1024;
const size_t MemoryControl::__40M__ = (size_t)40 * 1024 * 1024;
const size_t MemoryControl::__80M__ = (size_t)80 * 1024 * 1024;
const size_t MemoryControl::__160M__ = (size_t)160 * 1024 * 1024;
const size_t MemoryControl::__320M__ = (size_t)320 * 1024 * 1024;
const size_t MemoryControl::__640M__ = (size_t)640 * 1024 * 1024;
const size_t MemoryControl::__1280M__ = (size_t)1280 * 1024 * 1024;

MemoryControl::MemoryControl(size_t block_size):block_size(block_size) { }
MemoryControl::~MemoryControl() {
	for (auto bf: buffer) {
		CHECK( cudaFree(bf.first) );
	}
}

cudaError_t MemoryControl::deviceMalloc(void **devPtr, size_t size) {
	// 将 size 变为8的倍数
	if (size & 7) {
		size = ((size>>3)+1)<<3;
	}

	// 申请的空间大于了block的大小
	if (size > block_size) { 
		CHECK( cudaMalloc(devPtr, size) );
		CHECK( cudaDeviceSynchronize() );
		buffer.push_back(make_pair(*devPtr, *devPtr + size));
		return cudaSuccess;
	} 

	// 在之前以及分配的内存中寻找是否还有能够分配的
	for (size_t i = 0; i < buffer.size(); i++) {
		if (buffer[i].first + block_size >= buffer[i].second + size) {
			*devPtr = buffer[i].second;
			buffer[i].second += size;
			return cudaSuccess;
		}
	}

	// 之前的内存块都不够用, 直接向设备申请新的显存
	CHECK( cudaMalloc(devPtr, block_size) );
	CHECK( cudaDeviceSynchronize() );
	buffer.push_back(make_pair(*devPtr, *devPtr + size));

	return cudaSuccess;
}

cudaError_t MemoryControl::deviceFree(void *devPtr) {
	// do nothing
	return cudaSuccess;
}

cudaError_t MemoryControl::deviceMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
	CHECK( cudaMemcpy(dst, src, count, kind) );
	CHECK( cudaDeviceSynchronize() ); // 强行同步
}

void MemoryControl::Log() {
	for (int i = 0; i < buffer.size(); i++) {
		fprintf(stderr, "start: %lu, cur %lu\n", buffer[i].first, buffer[i].second);
	}
}

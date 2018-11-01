#ifndef __MEMORY_CONTROL_H__
#define __MEMORY_CONTROL_H__

#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
using namespace std;

#define CHECK(x) \
do {\
	cudaError_t err = (x);\
	if (err != cudaSuccess) {\
		 fprintf(stderr, "API error %s:%d Returned:%s\n", __FILE__, __LINE__, cudaGetErrorString(err));\
		 exit(1);\
	}\
} while(0)


// 好像使用cudaMalloc的次数不能很多, 不然会报错
// 所以实现了这个类型, 手动管理内存分配, 希望能消掉这个错误

class MemoryControl {
	private:
	size_t block_size;

	vector <pair<void*, void*> > buffer; 
	// first: buffer指针
	// second: 下一个准备分配的指针

	public:
	const static size_t __20M__;
	const static size_t __40M__; 
	const static size_t __80M__;
	const static size_t __160M__;
	const static size_t __320M__;
	const static size_t __640M__;
	const static size_t __1280M__;

	MemoryControl(size_t block_size);
	~MemoryControl();
	cudaError_t deviceMalloc(void **devPtr, size_t size);
	cudaError_t deviceFree(void *devPtr);
	cudaError_t deviceMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
	void Log(); // 方便在gdb中查看buffer中的值
};

#endif

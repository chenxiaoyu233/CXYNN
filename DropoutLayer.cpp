#include "DropoutLayer.h"
#include "cuda/kernels.h"

DropoutLayer::DropoutLayer(int channel, int row, int col, double prob, bool isTrain)
:Layer(channel, row, col), prob(prob), isTrain(isTrain) { 
#ifdef ENABLE_CUDA
	gpu_param_cnt = 2 * channel * row * col;
	CHECK( cudaMalloc(&gpu_param_ptr, sizeof(double) * gpu_param_cnt) );
#endif
}

DropoutLayer::~DropoutLayer() {
#ifdef ENABLE_CUDA
	CHECK( cudaFree(gpu_param_ptr) );
#endif
}

void DropoutLayer::connectLayer(Layer *Input) {
	assert(Input -> Channel() == channel);
	pair<int, int> sz = Input -> size();
	assert(sz.first == row && sz.second == col);
	FOR(c, 1, channel){ 
		SetAt(c); Input -> SetAt(c); // 设置分页
		FOR(x, 1, row) FOR(y, 1, col) {
			Insert(&(*Input)(x, y), &(*this)(x, y)); 
		}
	}
#ifdef ENABLE_CUDA
	//所有参数都已经生成, 开始转储
	cpu_param_ptr = new double* [gpu_param_cnt];
	for(int i = 0; i < gpu_param_cnt; i++) cpu_param_ptr[i] = paramPool[i];
	CHECK( cudaMemcpy(gpu_param_ptr, cpu_param_ptr, sizeof(double) * gpu_param_cnt, cudaMemcpyHostToDevice) );
	delete[] cpu_param_ptr;
	// 将所有的b相关的参数都置为0
	setBToZero(); 
#endif
}

void DropoutLayer::setBToZero() {
	int end = channel * row * col;
#ifdef ENABLE_CUDA
	kernel_vector_double_ptr_set_value(gpu_param_ptr, 0, 0, end);
#else
	for(int i = 0; i < end; i++) *(paramPool[i]) = 0;
#endif
}

void DropoutLayer::SetEdgeVal() {
	// 前面start个元素都是b对应的指针
	int start = channel * row * col;
#ifdef ENABLE_CUDA
	if(isTrain) kernel_vector_double_ptr_rand_zero_one(gpu_param_ptr, prob, start, gpu_param_cnt);
	else kernel_vector_double_ptr_set_value(gpu_param_ptr, prob, start, gpu_param_cnt);
#else
	if(isTrain)
		for(int i = start; i < paramPool.size(); i++) {
			double tmp = Tools.RandDouble(0.0, 1.0, 0.00001);
			if (tmp <= prob) *(paramPool[i]) = 1;
			else *(paramPool[i]) = 0;
		}
	else
		for(int i = start; i < paramPool.size(); i++) {
			*(paramPool[i]) = prob;
		}
#endif
}

void DropoutLayer::CollectParam(vector<double*> *param, vector<double*> *paramDel) {
	if (Input != NULL) {
		Input -> CollectParam(param, paramDel);
	}
}

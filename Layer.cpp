#include "Layer.h"
// cuda kernels
#ifdef ENABLE_CUDA
#include "cuda/kernels.h"
#endif


void Layer::layerInit() {
	paramPool.clear();
	paramDeltaPool.clear();
	Input = NULL;
	Output = NULL;
	//申请空间 + 参数收集
	FOR(cur, 1, channel){ SetAt(cur); //换面
		FOR(i, 1, row) FOR(j, 1, col) {
#ifdef ENABLE_CUDA
			CHECK( cudaMalloc(&(*this)(i, j).b, sizeof(double)) );
			CHECK( cudaMalloc(&(*this)(i, j).bDel, sizeof(double)) );
#else
			(*this)(i, j).b = new double;
			(*this)(i, j).bDel = new double;
#endif
			paramPool.push_back((*this)(i, j).b); //收集参数
			paramDeltaPool.push_back((*this)(i, j).bDel);
		}
	}
#ifdef ENABLE_CUDA
	// 添加idx, 便于在显存中重构网络.
	int count = channel * row * col - 1;
	FOR(i, 0, count) field[i].idx = i;
#endif
}

Layer::Layer(int row, int col): Matrix(row, col) { layerInit(); }

Layer::Layer(int channel, int row, int col)
	:Matrix(channel, row, col) { layerInit(); }

Layer::~Layer() {
	for(int i = 0; i < paramPool.size(); i++) {
#ifdef ENABLE_CUDA
		CHECK( cudaFree(paramPool[i]) );
#else
		delete paramPool[i];
#endif
	}
	for(int i = 0; i < paramDeltaPool.size(); i++) {
#ifdef ENABLE_CUDA
		CHECK( cudaFree(paramDeltaPool[i]) );
#else
		delete paramDeltaPool[i];
#endif
	}
}

void Layer::Insert(Neuron *a, Neuron *b) { // a -> b
	double *Wab, *dWab;
#ifdef ENABLE_CUDA
	CHECK( cudaMalloc(&Wab, sizeof(double)) );
	CHECK( cudaMalloc(&dWab, sizeof(double)) );
#else
	Wab = new double; dWab = new double;
#endif
	a -> Insert(OUTPUT, Wab, dWab, b);
	b -> Insert(INPUT, Wab, dWab, a); // 这里的两个w没有实际意义.
	paramPool.push_back(Wab); //收集参数
	paramDeltaPool.push_back(dWab);
}

void Layer::Insert(Neuron *a, Neuron *b, double *Wab, double *dWab) {
	a -> Insert(OUTPUT, Wab, dWab, b);
	b -> Insert(INPUT, Wab, dWab, a);
}

void Layer::InputLayer(Layer* Input) {
	this -> Input = Input;
	Input -> Output = this;
	this -> connectLayer(Input); // 防止虚函数踩坑
}

void Layer::SetActionFunc(
	double (*ActiveFunc) (double),
	double (*ActiveFuncDelta) (double)
) {
	FOR(c, 1, channel) { this -> SetAt(c);
		FOR(x, 1, row) FOR(y, 1, col) {
			(*this)(x, y).SetActionFunc(ActiveFunc, ActiveFuncDelta);
		}
	}
}

#ifndef ENABLE_CUDA
void Layer::UpdateForwardBegin() {
	FOR(c, 1, channel) { this -> SetAt(c);
		FOR(x, 1, row) FOR(y, 1, col) {
			(*this)(x, y).forwardBuffer[1] = 0;
		}
	}
	if (Output != NULL) {
		Output -> UpdateForward();
	}
}
#endif

void Layer::UpdateForwardBegin(Matrix<double> *other) {
	assert(row == (other->size()).first); //保证规模相同
	assert(col == (other->size()).second);
	assert(channel == other->Channel());
#ifdef ENABLE_CUDA
	double *gpu_buffer;
	CHECK( cudaMalloc(&gpu_buffer, sizeof(double) * channel * row * col) );
	CHECK( cudaMemcpy(gpu_buffer, other -> field, sizeof(double) * channel * row * col, cudaMemcpyHostToDevice) );
	kernel_layer_set_value(gpu_field, gpu_buffer, channel * row * col);
	CHECK( cudaFree(gpu_buffer) );
#else
	FOR(c, 1, channel) { this -> SetAt(c);
		FOR(x, 1, row) FOR(y, 1, col) {
			(*this)(x, y).SetValue((*other)(x, y));
		}
	}
#endif
	if (Output != NULL) {
		Output -> UpdateForward();
	}
}

#ifdef ENABLE_CUDA
void Layer::updateForward() {
	kernel_update_forward(gpu_field, channel * row * col);
}

void Layer::spreadBack() {
	kernel_spread_back(gpu_field, channel * row * col);
}
#else
void Layer::updateForward() {
	FOR(c, 1, channel) { this -> SetAt(c);
		FOR(x, 1, row) FOR(y, 1, col) {
			(*this)(x, y).UpdateBuffer();
		}
	}
}

void Layer::spreadBack() {
	FOR(c, 1, channel) { this -> SetAt(c);
		FOR(x, 1, row) FOR(y, 1, col) {
			(*this)(x, y).SpreadBack();
		}
	}
}
#endif

void Layer::UpdateForward() {
	this -> updateForward(); //内部
	if (Output != NULL) {
		Output -> UpdateForward();
	}
}

void Layer::SpreadBack() {
	if (Output != NULL) {
		this -> spreadBack(); //内部
	}
	if (Input != NULL) {
		Input -> SpreadBack();
	}
}

void Layer::CollectParam(vector<double*> *param, vector<double*> *paramDel) {
	assert(paramPool.size() == paramDeltaPool.size());
	for (int i = 0; i < paramPool.size(); i++) {
		param -> push_back(paramPool[i]);
		paramDel -> push_back(paramDeltaPool[i]);
	}
	if (Input != NULL) {
		Input -> CollectParam(param, paramDel);
	}
}

#ifdef ENABLE_CUDA
void Layer::syncFiber() {
	FOR(c, 1, channel) { this -> SetAt(c);
		FOR(x, 1, row) FOR(y, 1, col) {
			(*this)(x, y).SyncFiberInfo();
		}
	}
	mallocGpuMemory();
	syncMemFromHostToDevice();
}

void Layer::rebuildFiberOnGpu() {
	if(Input) kernel_rebuild_input_fiber(Input -> gpu_field, gpu_field, channel * row * col);
	if(Output) kernel_rebuild_output_fiber(Output -> gpu_field, gpu_field, channel * row * col);
}

void Layer::RebuildOnGPU() {
	syncFiber();
	if(Input != NULL) {
		Input -> RebuildOnGPU();
	}
	// 回溯时重建网络(保证空间已经申请好)
	rebuildFiberOnGpu();
}

void Layer::Sync_BackwardBuffer_to_bDel() {
	kernel_sync_BackwardBuffer_to_bDel(gpu_field, channel * row * col);
}
#endif

#ifndef __CXY_NEURON_NETWORK_H__
#define __CXY_NEURON_NETWORK_H__

using namespace std;

#include "Common.h"
#include "Fiber.h"
#include "Neuron.h"
#include "Layer.h"
#include "DenseLayer.h"
#include "ConvLayer.h"
#include "PoolLayer.h"
#include "MaxPoolLayer.h"
#include "Matrix.h"
#include "FuncAbstractor.h"
#include "Estimator.h"
#include "Optimizer.h"
#include "Predictor.h"

// cuda kernels
#ifdef ENABLE_CUDA
#include "cuda/kernels.h"
#endif

#endif 


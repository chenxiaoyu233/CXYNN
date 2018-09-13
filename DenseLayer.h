#ifndef __DENSE_LAYER_H__
#define __DENSE_LAYER_H__

#include "Common.h"
#include "Layer.h"

extern Tool Tools;

class DenseLayer: public Layer {
	protected:
	virtual void connectLayer(Layer* Input);

	public:
	DenseLayer(int row, int col);
};


#endif

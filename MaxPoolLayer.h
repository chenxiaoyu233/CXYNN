#ifndef __MAX_POOL_LAYER_H__
#define __MAX_POOL_LAYER_H__

#include "Common.h"
#include "PoolLayer.h"


class MaxPoolLayer: public PoolLayer {
	protected:
	void pushSpreadBack();
	virtual void updateForward();

	public:
	MaxPoolLayer(
		int channel,
		int row, int col,
		int coreRow = 2, int coreCol = 2,
		int stepRow = 2, int stepCol = 2,
		int padRow = 0, int padCol = 0
	);
	virtual void SpreadBack();
};

#endif

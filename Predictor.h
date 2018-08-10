#ifndef __PREDICTOR_H__
#define __PREDICTOR_H__

#include "Common.h"
#include "Optimizer.h"

// 用于在训练完成之后, 将网络实用化
class Predictor: public Optimizer {
	protected:
	vector< pair<int, Matrix<double>*> > expectPair; //所有的期望输出

	public:
	Predictor( //和Optimizer使用相同的构造函数
		FuncAbstractor* func, 
	    double step, 
	    int maxEpoch,
	    vector<Matrix<double>*> trainData,
	    vector<Matrix<double>*> trainLabel,
	    string filePath,
		int seed,
		double randL, double randR, double randEps,
		int miniBatchSize
	);
	int Classify(Matrix<double>* Input); //分类
	double Loss(Matrix<double>* Input, Matrix<double>* expect); //直接输出loss值
	void AddCase(int label, Matrix<double>* expect);
};

#endif

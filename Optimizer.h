#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__

#include "Common.h"
#include "Matrix.h"
#include "FuncAbstractor.h"


class Optimizer {
	protected:
	//超参数
	double step;
	int maxEpoch;
	int seed; //初始化参数使用的种子
	double randL, randR, randEps; //初始化参数使用的范围和精度
	int miniBatchSize; //SGD使用的 mini-batch 的大小

	//其他辅助参数
	int epoch;
	int saveStep;
	double meanLoss;
	string filePath; // 存储训练结果的文件
	FuncAbstractor *func;
	vector<Matrix<double>*> trainData;
	vector<Matrix<double>*> trainLabel;

	vector<double> direction;

	void TrainLoop();
	void LoadFromFile();
	virtual void Log();
	virtual void MainTrainMethod(int batchSize);

	public:
	Optimizer(FuncAbstractor* func, 
			  double step, 
			  int maxEpoch,
			  vector<Matrix<double>*> trainData,
			  vector<Matrix<double>*> trainLabel,
			  string filePath,
			  int seed,
			  double randL, double randR, double randEps,
			  int miniBatchSize);

	void TrainFromNothing();
	void TrainFromFile();
	void Save();
	void SetSaveStep(int step);

	//用于标准化输入数据
	void UnitlizeVector(vector<double> &data); //单位化向量
	void NormalizeData(Matrix<double>* data);
};

#endif

/*#include "CXYNeuronNetwork.h"

DenseLayer *input_layer;
DenseLayer *output_layer;

Estimator_QuadraticCost *estimator;

void build_network() {
	input_layer = new DenseLayer(1, 1);
	output_layer = new DenseLayer(1, 1);
	estimator = new Estimator_QuadraticCost(output_layer);

#ifdef ENABLE_CUDA
	input_layer -> SetActionFunc(kernel_Linear, kernel_LinearDel);
	output_layer -> SetActionFunc(kernel_Linear, kernel_LinearDel);
#else
	input_layer -> SetActionFunc(&(ActiveFunction::Linear), &(ActiveFunction::LinearDel));
	output_layer -> SetActionFunc(&(ActiveFunction::Linear), &(ActiveFunction::LinearDel));
#endif

	output_layer -> InputLayer(input_layer);
}

vector<Matrix<double>*> trainData;
vector<Matrix<double>*> trainLabel;

void genTrainData() {
	Matrix<double> *data = NULL, *label = NULL;
	for (int i = 0; i < 10000; i++) {
		data = new Matrix<double>(1, 1); 
		label = new Matrix<double>(1, 1);
		(*data)(1, 1) = Tools.RandDouble(-1.0, 1.0, 0.0001);
		(*label)(1, 1) = -(*data)(1, 1);
		trainData.push_back(data);
		trainLabel.push_back(label);
	}
}

void train() {
	FuncAbstractor func(input_layer, output_layer, estimator, 0.1);
	Optimizer opt(
		&func,
		0.001f,
		20000,
		trainData,
		trainLabel,
		"mnist/train_backup",
		2333,
		-0.05, 0.05, 0.00001,
		1000
	);

	opt.SetSaveStep(5);
	opt.TrainFromNothing();
	opt.Save();
}

void test() {
	FuncAbstractor func(input_layer, output_layer, estimator, 0.1);
	Predictor pre(
		&func,
		0.001f,
		20000,
		trainData,
		trainLabel,
		"mnist/train_backup",
		2333,
		-0.05, 0.05, 0.00001,
		200
	);

	Matrix<double> *data = new Matrix<double>(1, 1);
	Matrix<double> *label = new Matrix<double>(1, 1);
	for (int i = 0; i < 100; i++) {
		double s = Tools.RandDouble(-1.0, 1.0, 0.0001);
		(*data)(1, 1) = s; (*label)(1, 1) = -s;
		double loss = pre.Loss(data, label);
		printf("%.3f\n", loss);
	}
}

int main() {
#ifdef ENABLE_CUDA
	cuda_init();
#endif
	build_network();
	genTrainData();
	train();
	test();
	return 0;
}
*/
#include "CXYNeuronNetwork.h"
typedef unsigned char byte;

DenseLayer *Input;
ConvLayer *C1;
ConvLayer *S1;
ConvLayer *C2;
ConvLayer *S2;
DenseLayer *H1;
DropoutLayer *Dp1;
DenseLayer *Output;

Estimator_Softmax *estimator;

void build_network() {
	Input = new DenseLayer(28, 28); //
	C1 = new ConvLayer(32, 28, 28, 5, 5, 1, 1, 2, 2);
	S1 = new ConvLayer(32, 14, 14, 2, 2, 2, 2, 0, 0);
	C2 = new ConvLayer(16, 14, 14, 5, 5, 1, 1, 2, 2);
	S2 = new ConvLayer(16, 7, 7, 2, 2, 2, 2, 0, 0);
	H1 = new DenseLayer(1, 128);
	Dp1 = new DropoutLayer(1, 1, 128, 0.4, 0);
	Output = new DenseLayer(1, 10);

	estimator = new Estimator_Softmax(Output);

#ifdef ENABLE_CUDA
	Input -> SetActionFunc(kernel_Linear, kernel_LinearDel);
	C1 -> SetActionFunc(kernel_tanh, kernel_tanhDel);
	S1 -> SetActionFunc(kernel_Linear, kernel_LinearDel);
	C2 -> SetActionFunc(kernel_tanh, kernel_tanhDel);
	S2 -> SetActionFunc(kernel_Linear, kernel_LinearDel);
	H1 -> SetActionFunc(kernel_tanh, kernel_tanhDel);
	Dp1 -> SetActionFunc(kernel_Linear, kernel_LinearDel);
	Output -> SetActionFunc(kernel_Linear, kernel_LinearDel);
#else
	Input -> SetActionFunc(&ActiveFunction::Linear, &ActiveFunction::LinearDel);
	C1 -> SetActionFunc(&ActiveFunction::tanh, &ActiveFunction::tanhDel);
	S1 -> SetActionFunc(&ActiveFunction::Linear, &ActiveFunction::LinearDel);
	C2 -> SetActionFunc(&ActiveFunction::tanh, &ActiveFunction::tanhDel);
	S2 -> SetActionFunc(&ActiveFunction::Linear, &ActiveFunction::LinearDel);
	H1 -> SetActionFunc(&ActiveFunction::tanh, &ActiveFunction::tanhDel);
	Dp1 -> SetActionFunc(&ActiveFunction::Linear, &ActiveFunction::LinearDel);
	Output -> SetActionFunc(&ActiveFunction::Linear, &ActiveFunction::LinearDel);
#endif

	C1 -> InputLayer(Input);
	S1 -> InputLayer(C1);
	C2 -> InputLayer(S1);
	S2 -> InputLayer(C2);
	H1 -> InputLayer(S2);
	Dp1 -> InputLayer(H1);
	Output -> InputLayer(Dp1);
}

vector<Matrix<double>*> trainData;
vector<Matrix<double>*> trainLabel;

vector<Matrix<double>*> testData;
vector<Matrix<double>*> testLabel;

inline void readInt(int *buffer, FILE *in) { //Intel处理器读Mnist需要反转字节
	FOR(i, 0, 3) {
		*buffer <<= 8;
		fread(buffer, sizeof(byte), 1, in);
	}
}

void LogMatrix(Matrix<double>* mat) {
	mat -> Log();
}

void LogNeuron(Neuron *neu) {
	neu -> Log();
}

void read_data(vector<Matrix<double>*> &data, string filename) {
	puts("reading data ...");
	data.clear();
	FILE* in = fopen(filename.c_str(), "r");
	int magic, num, row, col;
	byte buffer;
	Matrix<double> *mat;
	readInt(&magic, in);
	readInt(&num, in);
	readInt(&row, in);
	readInt(&col, in);
	FOR(i, 1, num) {
		mat = new Matrix<double>(28, 28);
		FOR(x, 1, row) FOR(y, 1, col) {
			fread(&buffer, sizeof(byte), 1, in);
			(*mat)(x, y) = (double)buffer / 255.0 * 0.99 + 0.01;
		}
		data.push_back(mat);
	}
	fclose(in);
	puts("finish reading ...");
}

void read_label(vector<Matrix<double>*> &label, string filename) {
	puts("reading label ...");
	label.clear();
	FILE* in = fopen(filename.c_str(), "r");
	int magic, num;
	byte buffer;
	Matrix<double> *mat;
	readInt(&magic, in);
	readInt(&num, in);
	FOR(i, 1, num) {
		//mat = new Matrix<double>(1, 10);
		fread(&buffer, sizeof(byte), 1, in);
		//FOR(i, 1, 10) (*mat)(1, i) = 0.01f;
		//(*mat)(1, buffer+1) = 0.99f; //下标1 -> 10, 对应数字1 -> 9
		mat = new Matrix<double>(1, 1);
		(*mat)(1) = buffer + 1;
		label.push_back(mat);
	}
	fclose(in);
	puts("finish reading ...");
}

void read_train_data() {
	read_data(trainData, "mnist/train-images-idx3-ubyte");
}

void read_train_label() {
	read_label(trainLabel, "mnist/train-labels-idx1-ubyte");
}

void read_test_data() {
	read_data(testData, "mnist/t10k-images-idx3-ubyte");
}

void read_test_label() {
	read_label(testLabel, "mnist/t10k-labels-idx1-ubyte");
}

void train() {
	puts("initialize functionAbstractor");
	FuncAbstractor funcAbstractor(Input, Output, estimator, 0.1);
	puts("complete");


	puts("initialize optimizer");
	Optimizer optimizer(
		&funcAbstractor,
		0.05f,
		10000,
		trainData,
		trainLabel,
		"mnist/train_backup",
		2333,
		-0.10, 0.10, 0.0001,
		100
	);
	puts("complete");

	// add the dropout layer to the list
	optimizer.AddDropoutLayer(Dp1);

	optimizer.SetSaveStep(5);
	optimizer.TrainFromFile();
	//optimizer.TrainFromNothing();
	
	optimizer.Save(); //最后保存一下

}

void test() {
	puts("initialize functionAbstractor");
	FuncAbstractor funcAbstractor(Input, Output, estimator, 0.1);
	puts("complete");

	puts("initialize predictor");
	Predictor predictor(
		&funcAbstractor,
		0.05f,
		2000,
		trainData,
		trainLabel,
		"mnist/train_backup",
		2333,
		-0.10, 0.10, 0.0001,
		200
	);

	// add the dropout layer to the list
	predictor.AddDropoutLayer(Dp1);

	Matrix<double> *mat;
	FOR(i, 0, 9) {
		mat = new Matrix<double>(1, 1);
		(*mat)(1) = i+1;
		predictor.AddCase(i, mat); // 这里的内存泄漏了, 但是现在并不想管
	}
	puts("complete");

	int correct = 0;
	for (int i = 0; i < testData.size(); i++) {
		int ans = (*(testLabel[i]))(1);
		//FOR(j, 1, 10) if ((*(testLabel[i]))(1, j) > 0.5) ans = j;
		//printf("%d : %d\n", ans-1, predictor.Classify(testData[i]));
		if (ans-1 == predictor.Classify(testData[i])) correct += 1;
		if(i % 100 == 0) printf("%d/%d %d/%d\n", correct, i+1, i+1, testData.size());
	}
	printf("%d/%d\n", correct, testData.size());

}

int main() {
#ifdef ENABLE_CUDA
	cuda_init();
#endif
	build_network();
	read_train_data();
	read_train_label();
	read_test_data();
	read_test_label();

	//train();
	test();
	return 0;
}

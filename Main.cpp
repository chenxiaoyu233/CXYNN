/*#include "CXYNeuronNetwork.h"

Layer input_layer(1, 1);
Layer output_layer(1, 1);

Estimator_QuadraticCost estimator(&output_layer);

void build_network() {
	input_layer.SetActionFunc(&(ActiveFunction::Linear), &(ActiveFunction::LinearDel));
	output_layer.SetActionFunc(&(ActiveFunction::Linear), &(ActiveFunction::LinearDel));

	output_layer.InputFullConnect(&input_layer);
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
	FuncAbstractor func(&input_layer, &output_layer, &estimator);
	Optimizer opt(
		&func,
		0.001f,
		20000,
		trainData,
		trainLabel,
		"mnist/train_backup"
	);

	opt.SetSaveStep(5);
	opt.TrainFromNothing();
	opt.Save();
}

void test() {
	FuncAbstractor func(&input_layer, &output_layer, &estimator);
	Predictor pre(
		&func,
		0.001f,
		20000,
		trainData,
		trainLabel,
		"mnist/train_backup"
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
	build_network();
	genTrainData();
	train();
	test();
	return 0;
}*/

#include "CXYNeuronNetwork.h"
typedef unsigned char byte;

DenseLayer Input(28, 28); //
ConvLayer C1(6, 28, 28, 5, 5, 1, 1, 2, 2);
MaxPoolLayer S2(6, 14, 14, 2, 2, 2, 2, 0, 0);
ConvLayer C3(16, 10, 10, 5, 5, 1, 1, 0, 0);
MaxPoolLayer S4(16, 5, 5, 2, 2, 2, 2, 0, 0);
DenseLayer C5(1, 120);
DenseLayer F6(1, 84);
DenseLayer Output(1, 10);

Estimator_QuadraticCost estimator(&Output);

void build_network() {
	Input.SetActionFunc(&(ActiveFunction::Linear), &(ActiveFunction::LinearDel));
	C1.SetActionFunc(&(ActiveFunction::tanh), &(ActiveFunction::tanhDel));
	C3.SetActionFunc(&(ActiveFunction::Sigmoid), &(ActiveFunction::SigmoidDel));
	C5.SetActionFunc(&(ActiveFunction::tanh), &(ActiveFunction::tanhDel));
	F6.SetActionFunc(&(ActiveFunction::Sigmoid), &(ActiveFunction::SigmoidDel));
	Output.SetActionFunc(&(ActiveFunction::Linear), &(ActiveFunction::LinearDel));

	C1.InputLayer(&Input);
	S2.InputLayer(&C1);
	C3.InputLayer(&S2);
	S4.InputLayer(&C3);
	C5.InputLayer(&S4);
	F6.InputLayer(&C5);
	Output.InputLayer(&F6);
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
		mat = new Matrix<double>(1, 10);
		fread(&buffer, sizeof(byte), 1, in);
		FOR(i, 1, 10) (*mat)(1, i) = 0.01f;
		(*mat)(1, buffer+1) = 0.99f; //下标1 -> 10, 对应数字1 -> 9
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
	FuncAbstractor funcAbstractor(&Input, &Output, &estimator);
	puts("complete");


	puts("initialize optimizer");
	Optimizer optimizer(
		&funcAbstractor,
		0.6f,
		2000,
		trainData,
		trainLabel,
		"mnist/train_backup",
		2333,
		-0.02, 0.02, 0.0001,
		50
	);
	puts("complete");

	optimizer.SetSaveStep(5);
	optimizer.TrainFromNothing();
	
	optimizer.Save(); //最后保存一下

}

void test() {
	puts("initialize functionAbstractor");
	FuncAbstractor funcAbstractor(&Input, &Output, &estimator);
	puts("complete");

	puts("initialize predictor");
	Predictor predictor(
		&funcAbstractor,
		0.6f,
		2000,
		trainData,
		trainLabel,
		"mnist/train_backup",
		2333,
		-0.02, 0.02, 0.0001,
		50
	);

	Matrix<double> *mat;
	FOR(i, 0, 9) {
		mat = new Matrix<double>(1, 10);
		FOR(j, 1, 10) (*mat)(1, j) = 0.01f;
		(*mat)(1, i+1) = 0.99f;
		predictor.AddCase(i, mat); // 这里的内存泄漏了, 但是现在并不想管
	}
	puts("complete");

	int correct = 0;
	for (int i = 0; i < testData.size(); i++) {
		int ans = 0;
		FOR(j, 1, 10) if ((*(testLabel[i]))(1, j) > 0.5) ans = j;
		//printf("%d : %d\n", ans-1, predictor.Classify(testData[i]));
		if (ans-1 == predictor.Classify(testData[i])) correct += 1;
	}
	printf("%d/%d\n", correct, testData.size());

}

int main() {
	build_network();
	read_train_data();
	read_train_label();
	read_test_data();
	read_test_label();

	train();
	//test();
	return 0;
}


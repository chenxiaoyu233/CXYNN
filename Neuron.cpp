#include "Neuron.h"

Neuron::Neuron() {
	ActiveFunc = NULL;
	ActiveFuncDelta = NULL;
	input.clear(); output.clear();
	b = NULL; bDel = NULL;
	dropOutFlag = false;
	forwardBuffer[0] = forwardBuffer[1] = forwardBuffer[2] = 0;
	backwardBuffer = 0;
}

Neuron::Neuron(
	double (*ActiveFunc) (double),
	double (*ActiveFuncDelta) (double),
	double* b,
	double* bDel
): ActiveFunc(ActiveFunc), ActiveFuncDelta(ActiveFuncDelta), b(b), bDel(bDel){  
	input.clear(); output.clear();
	dropOutFlag = false;
	forwardBuffer[0] = forwardBuffer[1] = forwardBuffer[2] = 0;
	backwardBuffer = 0;
}

void Neuron::Insert(
	NeuronIO type, 
	double* weight,
	double* dWeight, 
	Neuron* neighbor
) {
	if (type == INPUT) input.push_back(Fiber(weight, dWeight, neighbor));
	if (type == OUTPUT) output.push_back(Fiber(weight, dWeight, neighbor));
}

void Neuron::SetValue(double x){
	x += *b;
	forwardBuffer[0] = x;
	forwardBuffer[1] = ActiveFunc(x);
	forwardBuffer[2] = ActiveFuncDelta(x);
}

void Neuron::SetActionFunc(
		double (*ActiveFunc) (double),
		double (*ActiveFuncDelta) (double)
) {
	this -> ActiveFunc = ActiveFunc;
	this -> ActiveFuncDelta = ActiveFuncDelta;
}

void Neuron::UpdateBuffer() {
	forwardBuffer[0] = 0;
	for (int i = 0; i < input.size(); i++) {
		forwardBuffer[0] += (input[i].neuron -> forwardBuffer[1]) * (*input[i].weight);
	}
	forwardBuffer[0] += *b;
	forwardBuffer[1] = ActiveFunc(forwardBuffer[0]);
	forwardBuffer[2] = ActiveFuncDelta(forwardBuffer[0]);
}

void Neuron::SpreadBack() {
	backwardBuffer = 0;
	for (int i = 0; i < output.size(); i++) {
		backwardBuffer += (output[i].neuron -> backwardBuffer) * (*output[i].weight) * forwardBuffer[2];
		*output[i].weightDel += (output[i].neuron -> backwardBuffer) * forwardBuffer[1];
		//同一个值可能被多次用到, 直接累加
	}
	*bDel += backwardBuffer;
}


void Neuron::Log() {
	printf("%.2lf %.2lf %.2lf\n", forwardBuffer[0], forwardBuffer[1], forwardBuffer[2]);
	printf("%.2lf %.2lf\n", backwardBuffer);
	for (int i = 0; i < input.size(); i++) {
		input[i].Log();
	}
	for (int i = 0; i < output.size(); i++) {
		output[i].Log();
	}
	printf("%.2lf %.2lf\n", b, bDel);
}

#include "Fiber.h"

Fiber::Fiber() { }

Fiber::Fiber(double* weight, double* weightDel, Neuron* neuron):
	weight(weight), weightDel(weightDel), neuron(neuron){ }

void Fiber::Log() {
	printf("Fiber: %.2f %.2f\n", *weight, *weightDel);
}

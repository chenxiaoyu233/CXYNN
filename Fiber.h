#ifndef __FIBER_H__
#define __FIBER_H__

#include "Common.h"


class Neuron;

class Fiber {
	public: //先把全部的访问权限开放
	double* weight; //边上的权重
	double* weightDel; //权值的偏导数
	Neuron* neuron; 

	Fiber();
	Fiber(double* weight, double* weightDel, Neuron* neuron);
	void Log();
};

#endif

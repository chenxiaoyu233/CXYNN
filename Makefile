CPP_FLAGS = -ggdb -c -DNDEBUG
COMPLIER = g++-7
CPP_FILES = $(shell ls *.cpp)
BASE = $(basename $(CPP_FILES))
OBJS = $(addsuffix .o, $(addprefix obj/, $(BASE)))
TARGET = debug/CXYNeuronNetwork

$(TARGET):$(OBJS)
	$(COMPLIER) -ggdb -o $(TARGET) $(OBJS)

obj/Main.o: Main.cpp CXYNeuronNetwork.h Common.h Fiber.h Neuron.h Layer.h \
 Matrix.h FuncAbstractor.h Estimator.h Optimizer.h
	$(COMPLIER) $(CPP_FLAGS) -o obj/Main.o Main.cpp

obj/Common.o: Common.cpp Common.h
	$(COMPLIER) $(CPP_FLAGS) -o obj/Common.o Common.cpp

obj/Estimator.o: Estimator.cpp Estimator.h Common.h Matrix.h Layer.h
	$(COMPLIER) $(CPP_FLAGS) -o obj/Estimator.o Estimator.cpp

obj/Fiber.o: Fiber.cpp Fiber.h Common.h
	$(COMPLIER) $(CPP_FLAGS) -o obj/Fiber.o Fiber.cpp

obj/FuncAbstractor.o: FuncAbstractor.cpp FuncAbstractor.h Common.h Layer.h \
 Matrix.h Neuron.h Estimator.h
	$(COMPLIER) $(CPP_FLAGS) -o obj/FuncAbstractor.o FuncAbstractor.cpp

obj/Layer.o: Layer.cpp Layer.h Common.h Matrix.h Neuron.h
	$(COMPLIER) $(CPP_FLAGS) -o obj/Layer.o Layer.cpp

obj/Matrix.o: Matrix.cpp Matrix.h Common.h
	$(COMPLIER) $(CPP_FLAGS) -o obj/Matrix.o Matrix.cpp

obj/Neuron.o: Neuron.cpp Neuron.h Common.h Fiber.h
	$(COMPLIER) $(CPP_FLAGS) -o obj/Neuron.o Neuron.cpp

obj/Optimizer.o: Optimizer.cpp Optimizer.h Common.h Matrix.h FuncAbstractor.h
	$(COMPLIER) $(CPP_FLAGS) -o obj/Optimizer.o Optimizer.cpp

obj/Predictor.o: Predictor.cpp Predictor.h Common.h Optimizer.h Matrix.h \
 FuncAbstractor.h Layer.h Neuron.h Fiber.h Estimator.h
	$(COMPLIER) $(CPP_FLAGS) -o obj/Predictor.o Predictor.cpp

obj/ConvLayer.o: ConvLayer.cpp ConvLayer.h Common.h Layer.h Matrix.h Neuron.h \
 Fiber.h
	$(COMPLIER) $(CPP_FLAGS) -o obj/ConvLayer.o ConvLayer.cpp

obj/PoolLayer.o: PoolLayer.cpp PoolLayer.h Common.h ConvLayer.h Layer.h \
 Matrix.h Neuron.h Fiber.h
	$(COMPLIER) $(CPP_FLAGS) -o obj/PoolLayer.o PoolLayer.cpp

obj/MaxPoolLayer.o: MaxPoolLayer.cpp MaxPoolLayer.h Common.h PoolLayer.h \
 ConvLayer.h Layer.h Matrix.h Neuron.h Fiber.h
	$(COMPLIER) $(CPP_FLAGS) -o obj/MaxPoolLayer.o MaxPoolLayer.cpp

obj/DenseLayer.o: DenseLayer.cpp DenseLayer.h Common.h Layer.h Matrix.h \
 Neuron.h Fiber.h
	$(COMPLIER) $(CPP_FLAGS) -o obj/DenseLayer.o DenseLayer.cpp

clean: 
	rm obj/*

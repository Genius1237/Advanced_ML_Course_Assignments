CC=g++ --std=c++14 -g
main: multi_layer_perceptron.o utilities.o matrix.o main.cpp
	$(CC) multi_layer_perceptron.o utilities.o matrix.o main.cpp

utilities.o: utilities.h utilities.cpp
	$(CC) -c utilities.cpp
	
multi_layer_perceptron.o: multi_layer_perceptron.h multi_layer_perceptron.cpp
	$(CC) -c multi_layer_perceptron.cpp

matrix.o : matrix.h matrix.cpp
	$(CC) -c matrix.cpp

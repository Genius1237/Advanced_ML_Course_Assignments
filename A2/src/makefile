CC=g++ --std=c++14 -g -O2
main: multi_layer_perceptron.o matrix.o main.cpp
	$(CC) multi_layer_perceptron.o matrix.o main.cpp
	
multi_layer_perceptron.o: multi_layer_perceptron.h multi_layer_perceptron.cpp matrix.o
	$(CC) -c multi_layer_perceptron.cpp

matrix.o : matrix.h matrix.cpp
	$(CC) -c matrix.cpp

clean:
	rm a.out multi_layer_perceptron.o matrix.o
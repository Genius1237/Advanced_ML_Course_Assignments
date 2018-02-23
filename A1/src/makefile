CC=g++ --std=c++14 -I./
main: classification_model.o utilities.o matrix.o
	$(CC) classification_model.o utilities.o matrix.o main.cpp

utilities.o: classification_model.o utilities.h utilities.cpp
	$(CC) -c utilities.cpp classification_model.o

classification_model.o: classification_model.h classification_model.cpp
	$(CC) -c classification_model.cpp

matrix.o : matrix.h matrix.cpp
	$(CC) -c matrix.cpp

CC=g++
CFLAGS=-shared
OS:=$(shell uname)
ifeq ($(OS), Darwin)
	CFLAGS+=-Wl,-install_name,flearner.so
endif
ifeq ($(OS), Linux)
	CFLAGS+=-Wl,-soname,flearner.so
endif

all: flearner.o
	$(CC) $(CFLAGS) -o flearner.so flearner.o

flearner.o: flearner.cpp
	$(CC) -c -fPIC flearner.cpp -o flearner.o

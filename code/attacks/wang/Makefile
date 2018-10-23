CC=g++
CFLAGS=-shared
OS:=$(shell uname)
ifeq ($(OS), Darwin)
	CFLAGS+=-Wl,-install_name,flearner.so
endif
ifeq ($(OS), Linux)
	CFLAGS+=-Wl,-soname,flearner.so
endif

all: flearner.so fextractor.py

flearner.so: flearner.o
	@echo Making library.
	$(CC) $(CFLAGS) -o flearner.so flearner.o

flearner.o: flearner.cpp
	$(CC) -c -fPIC flearner.cpp -o flearner.o

fextractor.py flearner.cpp:
	@echo Downloading.
	wget https://www.cse.ust.hk/~taow/wf/attacks.zip
	unzip attacks.zip attacks/flearner.cpp attacks/fextractor.py
	mv attacks/* .
	rmdir attacks
	rm attacks.zip
	@echo Patching.
	patch -p1 -s < wang.patch
	touch __init__.py

clean:
	rm -rf *.py *.o *.so flearner.cpp

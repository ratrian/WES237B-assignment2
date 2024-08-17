CC       = gcc
CFLAGS   = -g -Wall
INCFLAGS := -I../../helper_lib
LDFLAGS  := ../../helper_lib/helper_lib.a -lm

ifeq ($(shell uname -o), Darwin)
	LDFLAGS += -framework OpenCL
else ifeq ($(shell uname -o), GNU/Linux) # Assumes NVIDIA GPU
	LDFLAGS  += -L/usr/local/cuda/lib64 -lOpenCL
	INCFLAGS += -I/usr/local/cuda/include
else # Android
	LDFLAGS += -lOpenCL
endif

all: solution

solution: ../../helper_lib/helper_lib.a main.c
	$(CC) $(CFLAGS) -o $@ $^ $(INCFLAGS) $(LDFLAGS)

../../helper_lib/helper_lib.a: 
	cd ../../helper_lib; make

run: solution
	./solution ../Dataset_a/0/input0.raw ../Dataset_a/0/input1.raw ../Dataset_a/0/output.raw output.raw
	./solution ../Dataset_a/1/input0.raw ../Dataset_a/1/input1.raw ../Dataset_a/1/output.raw output.raw
	./solution ../Dataset_a/2/input0.raw ../Dataset_a/2/input1.raw ../Dataset_a/2/output.raw output.raw
	./solution ../Dataset_a/3/input0.raw ../Dataset_a/3/input1.raw ../Dataset_a/3/output.raw output.raw
	./solution ../Dataset_a/4/input0.raw ../Dataset_a/4/input1.raw ../Dataset_a/4/output.raw output.raw
	./solution ../Dataset_a/5/input0.raw ../Dataset_a/5/input1.raw ../Dataset_a/5/output.raw output.raw
	./solution ../Dataset_a/6/input0.raw ../Dataset_a/6/input1.raw ../Dataset_a/6/output.raw output.raw
	./solution ../Dataset_a/7/input0.raw ../Dataset_a/7/input1.raw ../Dataset_a/7/output.raw output.raw
	./solution ../Dataset_a/8/input0.raw ../Dataset_a/8/input1.raw ../Dataset_a/8/output.raw output.raw
	./solution ../Dataset_a/9/input0.raw ../Dataset_a/9/input1.raw ../Dataset_a/9/output.raw output.raw
	
clean: 
	rm -f solution
	cd ../../helper_lib; make clean
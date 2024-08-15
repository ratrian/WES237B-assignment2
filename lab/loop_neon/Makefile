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
	./solution ../Dataset_b/0/input0.raw ../Dataset_b/0/output.raw output.raw
	./solution ../Dataset_b/1/input0.raw ../Dataset_b/1/output.raw output.raw
	./solution ../Dataset_b/2/input0.raw ../Dataset_b/2/output.raw output.raw
	./solution ../Dataset_b/3/input0.raw ../Dataset_b/3/output.raw output.raw
	./solution ../Dataset_b/4/input0.raw ../Dataset_b/4/output.raw output.raw
	./solution ../Dataset_b/5/input0.raw ../Dataset_b/5/output.raw output.raw
	./solution ../Dataset_b/6/input0.raw ../Dataset_b/6/output.raw output.raw
	./solution ../Dataset_b/7/input0.raw ../Dataset_b/7/output.raw output.raw
	./solution ../Dataset_b/8/input0.raw ../Dataset_b/8/output.raw output.raw
	./solution ../Dataset_b/9/input0.raw ../Dataset_b/9/output.raw output.raw
	
clean: 
	rm -f solution
	cd ../../helper_lib; make clean
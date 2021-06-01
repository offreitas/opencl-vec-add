# Vector Addition
This project uses as template Alteras' 3D Finite Difference Stencil Computation. The original host code 
is the mainly modified file in order to perform just a vector addition.

## Table of contents
* [Explanation](#explanation)
* [Compilation](#compilation)
* [Running](#running)

## Explanation

The vectors parameters are the following as default:

```c++
// Vector sizes variables
int VEC_LEN = 64;
int VEC_SIZE = VEC_LEN*sizeof(float);
int VEC_NUM = 2;
```

About the variables:
* VEC_LEN: vector's length;
* VEC_SIZE: each vector's size;
* VEC_NUM: number of vectors.

The allocation and initialization of the vectors:
```c++
// Allocate host memory
h_data = (float *)malloc(VEC_SIZE*VEC_NUM*sizeof(float));
h_verify = (float *)malloc(VEC_SIZE*VEC_NUM*sizeof(float));
h_res = (float *)malloc(VEC_SIZE*VEC_NUM*sizeof(float));

// Initialize input
for(int i = 0; i < VEC_NUM; i++) {
    for(int j = 0; j < VEC_LEN; j++){
        *(h_data + i*VEC_LEN + j) = VEC_LEN*i;
        *(h_verify + i*VEC_LEN + j) = VEC_LEN*i;
    }
}
```

* h_data: array of vectors for the host;
* h_verify: copy from h_data to verify the results;
* h_res: array that stores the kernel's result.

The kernel code is as following:
```c++
__attribute__((task))
kernel void vector_sum(__global float* restrict a,
					   __global float* restrict b,
                	   __global float* restrict res, unsigned int length) {
	
	#pragma unroll
	for(int i = 0; i < length; i++)
		res[i] = a[i] + b[i];
}
```
## Compilation
### Compile Kernel for Emulation

```
aoc -march=emulator -fp-relaxed device/vector_sum.cl -o bin/vector_sum.aocx -legacy-emulator
```

### Compile Kernel to Generate Report

Go to the bin/ directory and run the following command line:
```
aoc -rtl ../device/vector_sum.cl
```

### Compile Host

```
make
```

## Running
### Running Emulation

```
CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 bin/host <arguments>
```
Available arguments:
* vec-num: number of vectors;
* vec-len: length of each vector.

***Example***: CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 bin/host -vec-num=4 -vec-len=16
Using these arguments set the host to compute the addition of 4 vectors of 16 elements.

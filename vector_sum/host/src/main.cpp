// Copyright (C) 2013-2019 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

///////////////////////////////////////////////////////////////////////////////////
// This OpenCL application executs a 3D Finite Difference Stencil Computation 
// on an Altera FPGA. The kernel is defined in a device/fd3d.cl file.  The Altera 
// Offline Compiler tool ('aoc') compiles the kernel source into a 'fd3d.aocx' 
// file containing a hardware programming image for the FPGA.  The host program 
// provides the contents of the .aocx file to the clCreateProgramWithBinary OpenCL
// API for runtime programming of the FPGA.
//
// When compiling this application, ensure that the Intel(R) FPGA SDK for OpenCL(TM)
// is properly installed.
///////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cstring>
#include <CL/opencl.h>
#include <CL/cl_ext_intelfpga.h>
#include "AOCLUtils/aocl_utils.h"

// Source the geometry restrictions (sizex and RADIUS)
#include "vector_sum_config.h"

using namespace aocl_utils;

// ACL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_kernel kernel = NULL;
static cl_program program = NULL;
static cl_int status = 0;

// Function prototypes
bool init();
bool vec_sum_gold(float *a, float *res);
void cleanup();

// Control whether the emulator should be used.
static bool use_emulator = false;

// Vector sizes variables
int VEC_LEN = 64;
int VEC_SIZE = VEC_LEN*sizeof(float);
int VEC_NUM = 2;

// Host memory buffers
float *h_data, *h_verify, *h_res;

// Device memory buffers
cl_mem d_dataA, d_dataB, d_res;

// Entry point.
int main(int argc, char **argv) {
	Options options(argc, argv);

	if(options.has("vec-len")){
		VEC_LEN = options.get<int>("vec-len");
		VEC_SIZE = VEC_LEN*sizeof(float);
	}

	if(options.has("vec-num")){
		if(options.get<int>("vec-num") >= 2)
			VEC_NUM = options.get<int>("vec-num");
		else 
			std::cout << "Number of vectors must be greater than or equal to 2" << std::endl;
	}

	if(!init()) {
		return false;
	}

	
	std::cout << "Vector length: " << VEC_LEN << std::endl;
	std::cout << "Vector size: " << VEC_SIZE << std::endl;
	std::cout << "Number of vectors: " << VEC_NUM << std::endl;

  	// Allocate host memory
	h_data = (float *)malloc(VEC_SIZE*VEC_NUM*sizeof(float));
	h_verify = (float *)malloc(VEC_SIZE*VEC_NUM*sizeof(float));
  	h_res = (float *)malloc(VEC_SIZE*VEC_NUM*sizeof(float));

	if (!(h_data && h_verify && h_res)) {
		printf("ERROR: Couldn't create host buffers\n");
		return false;
	}

	printf("Launching vector sum...\n");

	// Initialize input
	for(int i = 0; i < VEC_NUM; i++) {
		for(int j = 0; j < VEC_LEN; j++){
			*(h_data + i*VEC_LEN + j) = VEC_LEN*i;
			*(h_verify + i*VEC_LEN + j) = VEC_LEN*i;
		}
	}

	double time = 0.0;

	// Create device buffers - assign the buffers in different banks for more efficient
	// memory access
	for(int i = 1; i < VEC_NUM; i++){
		for(int j = 0; j < VEC_LEN; j++){
			d_dataA = clCreateBuffer(context, CL_MEM_READ_WRITE, VEC_SIZE, NULL, &status);
			checkError(status, "Failed to allocate input device buffer\n");
			d_dataB = clCreateBuffer(context, CL_MEM_READ_WRITE, VEC_SIZE, NULL, &status);
			checkError(status, "Failed to allocate output device buffer\n");
			d_res = clCreateBuffer(context, CL_MEM_WRITE_ONLY, VEC_SIZE, NULL, &status);
			checkError(status, "Failed to allocate constant device buffer\n");

			// Copy data from host to device
			status = clEnqueueWriteBuffer(queue, d_dataA, CL_TRUE, 0,
										  VEC_SIZE, (h_data + (i-1)*VEC_LEN + j),
										  0, NULL, NULL);
			checkError(status, "Failed to copy data to device");

			status = clEnqueueWriteBuffer(queue, d_dataB, CL_TRUE, 0,
										  VEC_SIZE, (h_data + i*VEC_LEN + j),
										  0, NULL, NULL);
			checkError(status, "Failed to copy data to device");

			// Set the kernel arguments
			status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_dataA);
			checkError(status, "Failed to set kernel arg 0");
			status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_dataB);
			checkError(status, "Failed to set kernel arg 1");
			status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_res);
			checkError(status, "Failed to set kernel arg 2");
			status = clSetKernelArg(kernel, 3, sizeof(int), (void *)&VEC_LEN);
			checkError(status, "Failed to set kernel arg 3");

			// Launch the kernel - we launch a single work item hence enqueue a task
			// Get the time_stepstamp to evaluate performance
			time = getCurrentTimestamp();
			status = clEnqueueTask(queue, kernel, 0, NULL, NULL);
			checkError(status, "Failed to launch kernel");
			clFinish(queue);

			status = clFinish(queue);
			checkError(status, "Failed to finish");

			// Copy results from device to host
			status = clEnqueueReadBuffer(queue, d_res, CL_TRUE, 0,
										 VEC_SIZE, (h_res + (i-1)*VEC_LEN + j),
										 0, NULL, NULL);
			checkError(status, "Failed to copy data from device");
		}
	}

	// Record execution time
	time = getCurrentTimestamp() - time;

	printf("\nProcessing time = %.4fms\n", (float)(time * 1E3));
	double gflops = ((double) VEC_SIZE / (time) * 1E-9);
	if (gflops < 0.001) {
		printf("Throughput = %.9f Gflops\n", gflops);
	} else {
		printf("Throughput = %.4f Gflops\n", gflops);
	}

	printf("\nVerifying data --> %s\n\n",
	vec_sum_gold(h_verify, h_res) ? "PASSED" : "FAILED");

	// Free the resources allocated
	cleanup();

	return 0;
}


/////// HELPER FUNCTIONS ///////

bool vec_sum_gold(float* a, float* res) {

	float gold_res;

	for(int i = 1; i < VEC_NUM; i++){
		for(int j = 0; j < VEC_LEN; j++){
			gold_res = *(a + (i-1)*VEC_LEN + j) + *(a + i*VEC_LEN + j);

			if(gold_res != *(res + (i-1)*VEC_LEN + j)) return false;
		}
	}

	return true;
}

bool init() {
	cl_int status;

	if(!setCwdToExeDir()) {
		return false;
	}

	// Get the OpenCL platform.
	platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
	
	if(platform == NULL) {
		printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform\n");
		return false;
	}

	// Query the available OpenCL devices.
	scoped_array<cl_device_id> devices;
	cl_uint num_devices;

	devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));

	// We'll just use the first device.
	device = devices[0];

	// Create the context.
	context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &status);
	checkError(status, "Failed to create context");

	// Create the command queue.
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command queue");

	// Create the program.
	std::string binary_file = getBoardBinaryFile("vector_sum", device);
	printf("Using AOCX: %s\n\n", binary_file.c_str());
	program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

	// Build the program that was just created.
	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	checkError(status, "Failed to build program");

	// Create the kernel - name passed in here must match kernel name in the
	// original CL file, that was compiled into an AOCX file using the AOC tool
	const char *kernel_name = "vector_sum";  // Kernel name, as defined in the CL file
	kernel = clCreateKernel(program, kernel_name, &status);
	checkError(status, "Failed to create kernel");

  	return true;
}

// Free the resources allocated during initialization
void cleanup() {
	if(kernel) 
		clReleaseKernel(kernel);  
	if(program) 
		clReleaseProgram(program);
	if(queue) 
		clReleaseCommandQueue(queue);
	if(h_verify)
		alignedFree(h_verify);
	if(h_data)
		alignedFree(h_data);
	if(h_res)
		alignedFree(h_res);
	if(d_dataA)
		clReleaseMemObject(d_dataA);
	if(d_dataB)
		clReleaseMemObject(d_dataB);
	if(d_res)
		clReleaseMemObject(d_res);
	if(context) 
		clReleaseContext(context);
}




__attribute__((task))
kernel void vector_sum(__global float* restrict a,
		       __global float* restrict b,
                       __global float* restrict res, unsigned int length) {
	
	#pragma unroll
	for(int i = 0; i < length; i++)
		res[i] = a[i] + b[i];
}

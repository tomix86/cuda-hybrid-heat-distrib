#include <cstdio>
#include <sstream>
#include <stdexcept>
#include "kernel.hpp"
#include "Mesh.hpp"

static std::string to_string(cudaError_t error) {
	char buf[256];
	snprintf(buf, 256, "%d", error);
	return buf;
}


class CudaError : public std::runtime_error {
public:
	CudaError(std::string source, cudaError_t errorCode) :
		std::runtime_error( source + ": code" + to_string(errorCode) + ": " + cudaGetErrorString(errorCode) ) {
	}
};

#define checkCudaErrors( val ) checkError( ( val ), #val, __FILE__, __LINE__ )
void checkError(cudaError_t result, const char* calledFunc,  const char* file, int line) {
	if (result) {
		std::ostringstream ss;
		ss << file << ": " << line << " {" << calledFunc << '}';

		throw CudaError(ss.str(), result);
	}
}

__global__ void meshUpdateKernel(float* mesh_in, float* mesh_out, size_t pitch, unsigned size) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if ( x > 0 && x < size - 1 && y > 0 && y < size - 1) {
		const float t_left = *getElem(mesh_in, pitch, y, x - 1);
		const float t_right = *getElem(mesh_in, pitch, y, x + 1);
		const float t_top = *getElem(mesh_in, pitch, y - 1, x); 
		const float t_bottom = *getElem(mesh_in, pitch, y + 1, x);

		const float newTemperature = (t_left + t_right + t_top + t_bottom) / 4;
		
		*getElem(mesh_out, pitch, y, x) = newTemperature;
	}
}


// optimal block size is 128,1,1
__global__ void meshUpdateKernel_opt1(float *mesh_in, float *mesh_out, size_t pitch, unsigned size) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	//TODO: switch to dynamic shared memory
	__shared__ float shared[3][128 + 2];
/*
	if (threadIdx.x == 0) {
		if (x > 0) {
			shared[1][0] = *getElem(mesh_in, pitch, y, x - 1);
		}
		else {
			shared[1][1] = *getElem(mesh_in, pitch, y, x);
		}
	}

	if (threadIdx.x == blockDim.x - 1) {
		if (x < size - 1) {
			shared[1][blockDim.x + 1] = *getElem(mesh_in, pitch, y, x + 1);
		}
		else {
			const auto pos = size - blockIdx.x * blockDim.x;
			shared[1][pos] = *getElem(mesh_in, pitch, y, blockIdx.x * blockDim.x + pos - 1);
		}
	}*/
	
	if (x > 0 && x < size - 1 && y > 0 && y < size - 1) {
//		shared[1][threadIdx.x + 1 - 1] = *getElem(mesh_in, pitch, y, x - 1);
//		shared[1][threadIdx.x + 1 + 1] = *getElem(mesh_in, pitch, y, x + 1);
		shared[0][threadIdx.x + 1] = *getElem(mesh_in, pitch, y-1, x);
		shared[1][threadIdx.x + 1] = *getElem(mesh_in, pitch, y, x);
		shared[2][threadIdx.x + 1] = *getElem(mesh_in, pitch, y+1, x);

		__syncthreads();

		const float t_l = shared[1][threadIdx.x + 1 - 1];
		const float t_r = shared[1][threadIdx.x + 1 + 1];
		const float t_t = shared[0][threadIdx.x + 1];
		const float t_b = shared[2][threadIdx.x + 1];

		const float newTemperature = (t_l + t_r + t_b + t_t) / 4;

//		printf("[%d,%d]: {%f;%f;%f;%f}: %f\n", x, y, t_l, t_r, t_t, t_b, newTemperature);

		*getElem(mesh_out, pitch, y, x) = newTemperature;
	}
}


void cuda() {
	size_t pitch;
	float *temperature = allocMeshLinear(pitch);
	size_t d_pitch;
	float *d_temperature_in, *d_temperature_out;

	try {
		checkCudaErrors(cudaMallocPitch(&d_temperature_in, &d_pitch, MESH_SIZE_EXTENDED * sizeof(float), MESH_SIZE_EXTENDED));
		checkCudaErrors(cudaMallocPitch(&d_temperature_out, &d_pitch, MESH_SIZE_EXTENDED * sizeof(float), MESH_SIZE_EXTENDED));
	}
	catch (CudaError& err) {
		std::cout << err.what() << std::endl;
		return;
	}

	try {
		SimpleTimer t( "CUDA implementation" );
		dim3 blockSize(BLOCK_DIM_X, BLOCK_DIM_Y);
		unsigned computedGridDimX = (MESH_SIZE_EXTENDED + blockSize.x - 1) / blockSize.x;
		unsigned computedGridDimY = (MESH_SIZE_EXTENDED + blockSize.y - 1) / blockSize.y;
		dim3 gridSize(computedGridDimX, computedGridDimY);

		checkCudaErrors(cudaMemcpy2D(d_temperature_in, d_pitch, temperature, pitch, MESH_SIZE_EXTENDED * sizeof(float), MESH_SIZE_EXTENDED, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy2D(d_temperature_out, d_pitch, d_temperature_in, d_pitch, MESH_SIZE_EXTENDED * sizeof(float), MESH_SIZE_EXTENDED, cudaMemcpyDeviceToDevice));

		for (int step = 0; step < STEPS; ++step) {
			meshUpdateKernel << < gridSize, blockSize >> > (d_temperature_in, d_temperature_out, d_pitch, MESH_SIZE_EXTENDED);
			checkCudaErrors(cudaGetLastError()); // Check for any errors launching the kernel
			checkCudaErrors(cudaDeviceSynchronize());// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
			std::swap(d_temperature_in, d_temperature_out);
		}

		checkCudaErrors(cudaMemcpy2D(temperature, pitch, d_temperature_in, d_pitch, MESH_SIZE_EXTENDED * sizeof(float), MESH_SIZE_EXTENDED, cudaMemcpyDeviceToHost));
	}
	catch (CudaError& err) {
		std::cout << err.what() << std::endl;
	}

	validateResults(temperature, pitch);

	delete[] temperature;
	try {
		checkCudaErrors(cudaFree(d_temperature_in));
		checkCudaErrors(cudaFree(d_temperature_out));
	}
	catch (CudaError& err) {
		std::cout << err.what() << std::endl;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	checkCudaErrors(cudaDeviceReset());
}

//
//
// Hybrid implementation
//
//

__global__ void meshUpdateKernel_hybrid(float* mesh_in, float* mesh_out, size_t pitch, unsigned size_x, unsigned size_y) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x > 0 && x < size_x - 1 && y > 0 && y < size_y - 1) {
		const float t_left = *getElem(mesh_in, pitch, y, x - 1);
		const float t_right = *getElem(mesh_in, pitch, y, x + 1);
		const float t_top = *getElem(mesh_in, pitch, y - 1, x);
		const float t_bottom = *getElem(mesh_in, pitch, y + 1, x);

		const float newTemperature = (t_left + t_right + t_top + t_bottom) / 4;

		*getElem(mesh_out, pitch, y, x) = newTemperature;

	//	printf("[%d,%d]: {%f;%f;%f;%f}: %f\n", x, y, t_left, t_right, t_top, t_bottom, newTemperature);
	}
}


HybridCuda::HybridCuda(size_t divisionPoint, size_t pitch, int deviceId) :
DIVISION_POINT(divisionPoint),
pitch(pitch),
deviceId(deviceId) {
	part = (deviceId == 0 ? BOTTOM : TOP);

	if (part == BOTTOM) {
		allocNumRows = MESH_SIZE_EXTENDED - (DIVISION_POINT - 1);
	}
	else {
		allocNumRows = DIVISION_POINT + 1;
	}

	setDevice();

	try {
		checkCudaErrors(cudaMallocPitch(&d_temperature_in, &d_pitch, MESH_SIZE_EXTENDED * sizeof(float), allocNumRows));
		checkCudaErrors(cudaMallocPitch(&d_temperature_out, &d_pitch, MESH_SIZE_EXTENDED * sizeof(float), allocNumRows));
	}
	catch (CudaError& err) {
		std::cout << err.what() << std::endl;
		return;
	}
}

HybridCuda::~HybridCuda() {
	setDevice();

	try {
		checkCudaErrors(cudaFree(d_temperature_in));
		checkCudaErrors(cudaFree(d_temperature_out));
	}
	catch (CudaError& err) {
		std::cout << err.what() << std::endl;
	}
}

//TODO: overlap computation and communication - launch separate streams
void HybridCuda::launchCompute(float* temperature_in) {
	setDevice();

	try {
		dim3 blockSize(BLOCK_DIM_X, BLOCK_DIM_Y);
		unsigned computedGridDimX = (MESH_SIZE_EXTENDED + blockSize.x - 1) / blockSize.x;
		unsigned computedGridDimY = (allocNumRows + blockSize.y - 1) / blockSize.y;
		dim3 gridSize(computedGridDimX, computedGridDimY);

		float* srcPtr;
		if (part == BOTTOM) {
			srcPtr = reinterpret_cast<float*>(reinterpret_cast<char*>(temperature_in) + (DIVISION_POINT - 1) * pitch);
		}
		else {
			srcPtr = temperature_in;
		}

		checkCudaErrors(cudaMemcpy2D(d_temperature_in, d_pitch, srcPtr, pitch, MESH_SIZE_EXTENDED * sizeof(float), 1, cudaMemcpyHostToDevice));

		meshUpdateKernel_hybrid<<< gridSize, blockSize >>> (d_temperature_in, d_temperature_out, d_pitch, MESH_SIZE_EXTENDED, allocNumRows);
	//	checkCudaErrors(cudaGetLastError()); // Check for any errors launching the kernel
	}
	catch (CudaError& err) {
		std::cout << err.what() << std::endl;
	}
}

void HybridCuda::finalizeCompute(float* temperature_out) {
	setDevice();

	try {
		checkCudaErrors(cudaDeviceSynchronize());// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.

		float *srcPtr, *dstPtr;
		if (part == BOTTOM) {
			dstPtr = reinterpret_cast<float*>(reinterpret_cast<char*>(temperature_out) + DIVISION_POINT * pitch);
			srcPtr = reinterpret_cast<float*>(reinterpret_cast<char*>(d_temperature_out) + d_pitch);
		}
		else {
			dstPtr = reinterpret_cast<float*>(reinterpret_cast<char*>(temperature_out) + (DIVISION_POINT + 1) * pitch);
			srcPtr = reinterpret_cast<float*>(reinterpret_cast<char*>(d_temperature_out) + (allocNumRows - 1) * pitch); //TODO: verify
		}


		checkCudaErrors(cudaMemcpy2D(dstPtr, pitch, srcPtr, d_pitch, MESH_SIZE_EXTENDED * sizeof(float), 1, cudaMemcpyDeviceToHost));

		std::swap(d_temperature_in, d_temperature_out);
	}
	catch (CudaError& err) {
		std::cout << err.what() << std::endl;
	}
}

void HybridCuda::copyInitial(float* temperature_in) {
	setDevice();

	float* srcPtr;
	if (part == BOTTOM) {
		srcPtr = reinterpret_cast<float*>(reinterpret_cast<char*>(temperature_in) + (DIVISION_POINT - 1) * pitch);
	}
	else {
		srcPtr = temperature_in;
	}
	checkCudaErrors(cudaMemcpy2D(d_temperature_in, d_pitch, srcPtr, pitch, MESH_SIZE_EXTENDED * sizeof(float), allocNumRows, cudaMemcpyHostToDevice));

	//TODO: remove this copy - copy only last row
	checkCudaErrors(cudaMemcpy2D(d_temperature_out, d_pitch, d_temperature_in, d_pitch, MESH_SIZE_EXTENDED * sizeof(float), allocNumRows, cudaMemcpyDeviceToDevice));
}

void HybridCuda::copyFinal(float* temperature_out) {
	setDevice();

	float *srcPtr, *dstPtr;
	if (part == BOTTOM) {
		dstPtr = reinterpret_cast<float*>(reinterpret_cast<char*>(temperature_out) + DIVISION_POINT * pitch);
		srcPtr = reinterpret_cast<float*>(reinterpret_cast<char*>(d_temperature_in) + d_pitch);
	}
	else {
		dstPtr = temperature_out;
		srcPtr = d_temperature_in;
	}

	checkCudaErrors(cudaMemcpy2D(dstPtr, pitch, srcPtr, d_pitch, MESH_SIZE_EXTENDED * sizeof(float), allocNumRows - 1, cudaMemcpyDeviceToHost));
}

void HybridCuda::setDevice() {
	cudaSetDevice(deviceId);
}

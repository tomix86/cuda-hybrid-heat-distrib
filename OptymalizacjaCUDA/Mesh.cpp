#include "Mesh.hpp"
//externs
size_t STEPS;
size_t MESH_SIZE;
size_t MESH_SIZE_EXTENDED;
int BLOCK_DIM_X;
int BLOCK_DIM_Y;


float Mesh::ENVIRONMENT_TEMP = 50.f;
float Mesh::INITIAL_TEMP = 25.f;
Mesh::ContainerType Mesh::temperature;

bool VALIDATE_RESULTS = false;

void setValidateResults(bool value) {
	VALIDATE_RESULTS = value;
}

float** allocMesh() {
	float** output;

	try {
		output = new float*[MESH_SIZE_EXTENDED];
		for (int i = 0; i < MESH_SIZE_EXTENDED; ++i) {
			output[i] = new float[MESH_SIZE_EXTENDED];
		}

		// Set temperature for guardian fields
		for (int i = 0; i < MESH_SIZE_EXTENDED; ++i) {
			for (int j = 0; j < MESH_SIZE_EXTENDED; ++j) {
				if (i == 0 || i == MESH_SIZE_EXTENDED - 1 || j == 0 || j == MESH_SIZE_EXTENDED - 1) {
					output[i][j] = Mesh::ENVIRONMENT_TEMP;
				}
				else {
					output[i][j] = Mesh::INITIAL_TEMP;
				}
			}
		}

		return output;
	}
	catch (std::exception ex) {
		std::cout << ex.what() << std::endl;
		exit(-1);
	}
}

float* allocMeshLinear(size_t& pitch) {
	pitch = MESH_SIZE_EXTENDED * sizeof(float);
	float* output;

	try {
		output = new float[MESH_SIZE_EXTENDED * MESH_SIZE_EXTENDED];

		// Set temperature for guardian fields
		for (int i = 0; i < MESH_SIZE_EXTENDED; ++i) {
			for (int j = 0; j < MESH_SIZE_EXTENDED; ++j) {
				if (i == 0 || i == MESH_SIZE_EXTENDED - 1 || j == 0 || j == MESH_SIZE_EXTENDED - 1) {
					*getElem(output, pitch, i, j) = Mesh::ENVIRONMENT_TEMP;
				}
				else {
					*getElem(output, pitch, i, j) = Mesh::INITIAL_TEMP;
				}
			}
		}

		return output;
	}
	catch (std::exception ex) {
		std::cout << ex.what() << std::endl;
		exit(-1);
	}
}

void freeMesh(float** mesh) {
	for (int i = 0; i < MESH_SIZE_EXTENDED; ++i) {
		delete[] mesh[i];
	}
	delete[] mesh;

}

void validateResults(float** input) {
	if (VALIDATE_RESULTS) {
		for (int i = 0; i < MESH_SIZE; ++i) {
			for (int j = 0; j < MESH_SIZE; ++j) {
				const float threshold = .001f;
				if (fabs(Mesh::temperature[i][j] - input[i + 1][j + 1]) > threshold) {
					std::cout << "Results are different for optimized version!\n"
						<< "Mismatch for [" << i << "][" << j << "]: " << Mesh::temperature[i][j] << " vs " << input[i + 1][j + 1] << std::endl;
					return;
				}
			}
		}
	}
}

void validateResults(float* input, size_t pitch) {
	if (VALIDATE_RESULTS) {
		for (int i = 0; i < MESH_SIZE; ++i) {
			for (int j = 0; j < MESH_SIZE; ++j) {
				const float threshold = .001f;
				const auto value = *getElem(input, pitch, i + 1, j + 1);
				if (fabs(Mesh::temperature[i][j] - value) > threshold) {
					std::cout << "Results are different for optimized version!\n"
						<< "Mismatch for [" << i << "][" << j << "]: " << Mesh::temperature[i][j] << " vs " << value << std::endl;
					return;
				}
			}
		}
	}
}
ROOT_DIR=OptymalizacjaCUDA/
INT_DIR=obj/
SRC_DIR=$(ROOT_DIR)
OUT_DIR=$(ROOT_DIR)bin/

INC_DIR=$(SRC_DIR) /usr/local/cuda/include/
LIB_DIR=/usr/local/cuda/lib64/

#TODO: specyfikowanie przy budowaniu czy debug czy release

CC_EXTENSIVE_ERRORS= -Werror -Wfatal-errors -Wall -Wextra -pedantic -pedantic-errors \
-Wswitch-default -Wswitch-enum -Wcast-align -Wpointer-arith -Wstrict-overflow=5 -Winline \
-Wundef -Wcast-qual -Wshadow -Wunreachable-code -Wwrite-strings -Wuninitialized -Wlogical-op \
-Wfloat-equal -Wstrict-aliasing=2 -Wredundant-decls -fno-omit-frame-pointer -ffloat-store \
-fno-common -fstrict-aliasing -Winit-self

#CC=g++
CC=/opt/intel-parallel-studio/bin/icc
CC_DEBUG= -Og -ggdb3 -fopenmp
#CC_RELEASE= -O3 -fopenmp
CC_RELEASE= -03 -openmp -vec-report3
CCFLAGS=-std=c++0x $(CC_RELEASE)

NVCC=nvcc
NVCC_SUPPRESSED_WARNINGS="--diag_suppress=virtual_function_decl_hidden"
NCCC_DEBUG=--source-in-ptx -lineinfo
NCCC_RELEASE=--use_fast_math -maxregcount=32
NVCCFLAGS=-std=c++11 -arch=sm_20 -rdc=true -Xcompiler -fPIC -Xcudafe $(NVCC_SUPPRESSED_WARNINGS) -Xptxas -v $(NVCC_RELEASE)

NVCCLIBS=-lcudart
LIBS=

NVCCOBJECTS=kernel.o
OBJECTS=main.o Mesh.o $(NVCCOBJECTS)

all: create_dirs cuda-heat

create_dirs:
	@mkdir -p $(OUT_DIR)
	@mkdir -p $(INT_DIR)

clean:
	@rm -rf $(OUT_DIR)
	@rm -rf $(INT_DIR)

kernels: create_dirs GPUComputationPerformer.o

# If you do not understand the use of: %, $<, $@ please refer to the following pages:
# https://stackoverflow.com/questions/7404444/what-does-a-percent-symbol-do-in-a-makefile
# https://www.gnu.org/software/make/manual/make.html#Static-Usage

# Two linking steps are required because of relocatable device code, detailed information and explanation can be found here:
# https://stackoverflow.com/questions/22115197/dynamic-parallelism-undefined-reference-to-cudaregisterlinkedbinary-linking
# http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#using-separate-compilation-in-cuda
cuda-heat: $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) -dlink $(patsubst %,$(INT_DIR)%, $(NVCCOBJECTS)) -o $(INT_DIR)LINKED_DEVICE_CODE $(NVCCLIBS)
	$(CC) -fopenmp $(patsubst %,$(INT_DIR)%, $(OBJECTS)) $(INT_DIR)LINKED_DEVICE_CODE -o $(OUT_DIR)cuda-heat $(patsubst %,-L%, $(LIB_DIR)) $(LIBS) $(NVCCLIBS)
#	cp -a $(LOCAL_LIBS_DIR). $(OUT_DIR)
#	cp -a $(RES_DIR). $(OUT_DIR)res/

#files compiled with the C++ compiler
%.o: $(SRC_DIR)%.cpp
	$(CC) -c $(CCFLAGS) $< -o $(INT_DIR)$@ $(patsubst %,-I%, $(INC_DIR))

#files compiled with the Nvidia C++ compiler
%.o: $(SRC_DIR)%.cu
	$(NVCC) -c $(NVCCFLAGS) $< -o $(INT_DIR)$@ $(patsubst %,-I%, $(INC_DIR))

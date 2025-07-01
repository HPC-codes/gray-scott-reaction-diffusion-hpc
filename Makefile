todavia no lo pruebes, lo pongo aqui para copiarlo ahorita y seguir las pruebas
# === Compiladores ===
CC = g++
NVCC = nvcc
MPICC = mpicxx

# === Directorios ===
BASE_DIR = /content/gray-scott-reaction-diffusion-hpc
BIN_DIR = $(BASE_DIR)/bin
DATA_DIR = $(BASE_DIR)/data
SRC_DIR = $(BASE_DIR)/src
SCRIPTS_DIR = $(BASE_DIR)/scripts

# === Flags de compilación ===
CFLAGS = -Wall -O3
NVCCFLAGS = -O3 --default-stream per-thread -arch=sm_75
MPIFLAGS = -Wall -O3
OMPFLAGS = -fopenmp -Wall -O3 -march=native -ffast-math -funroll-loops

# === Ejecutables ===
SERIAL_EXE = $(BIN_DIR)/gray_scott_serial
MPI_EXE = $(BIN_DIR)/gray_scott_mpi
CUDA_EXE = $(BIN_DIR)/gray_scott_cuda
OMP_EXE = $(BIN_DIR)/gray_scott_omp

# === Reglas principales ===
all: serial mpi cuda omp

serial: $(SERIAL_EXE)
mpi: $(MPI_EXE)
cuda: $(CUDA_EXE)
omp: $(OMP_EXE)

# === Reglas de compilación ===
$(SERIAL_EXE): $(SRC_DIR)/serial.cpp
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $< -o $@

$(MPI_EXE): $(SRC_DIR)/mpi.cpp
	@mkdir -p $(BIN_DIR)
	$(MPICC) $(MPIFLAGS) $< -o $@

$(CUDA_EXE): $(SRC_DIR)/cuda.cu
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(OMP_EXE): $(SRC_DIR)/OMP.cpp
	@mkdir -p $(BIN_DIR)
	$(CC) $(OMPFLAGS) $< -o $@
	@echo "Compilación OpenMP optimizada completada - Lista para Colab"

# === Reglas de ejecución individuales ===
run_serial: serial
	@mkdir -p $(DATA_DIR)
	@echo "Ejecutando versión Serial..."
	@$(SERIAL_EXE)

run_mpi: mpi
	@mkdir -p $(DATA_DIR)
	@echo "Ejecutando versión MPI..."
	@mpirun -np 4 $(MPI_EXE)

run_cuda: cuda
	@mkdir -p $(DATA_DIR)
	@echo "Ejecutando versión CUDA..."
	@$(CUDA_EXE)

run_omp: omp
	@mkdir -p $(DATA_DIR)
	@echo "Ejecutando versión OpenMP optimizada..."
	@OMP_NUM_THREADS=$(shell nproc) OMP_PROC_BIND=close OMP_PLACES=cores $(OMP_EXE)

# === Ejecución interactiva ===
run:
	@echo "=== Selecciona la versión a ejecutar ==="
	@echo "1) Serial"
	@echo "2) MPI"
	@echo "3) CUDA"
	@echo "4) OpenMP (Optimizado para Colab)"
	@read -p "Opción [1-4]: " opt; \
	if [ $$opt = "1" ]; then \
		$(MAKE) run_serial; \
	elif [ $$opt = "2" ]; then \
		$(MAKE) run_mpi; \
	elif [ $$opt = "3" ]; then \
		$(MAKE) run_cuda; \
	elif [ $$opt = "4" ]; then \
		$(MAKE) run_omp; \
	else \
		echo "Opción inválida"; exit 1; \
	fi

# === Opciones adicionales para OpenMP ===
run_omp_profile: omp
	@mkdir -p $(DATA_DIR)
	@echo "Ejecutando versión OpenMP con profiling..."
	@$(CC) $(OMPFLAGS) -pg $(SRC_DIR)/OMP.cpp -o $(OMP_EXE)_profile
	@OMP_NUM_THREADS=$(shell nproc) $(OMP_EXE)_profile
	@gprof $(OMP_EXE)_profile gmon.out > omp_profile_results.txt
	@echo "Resultados del profiling guardados en omp_profile_results.txt"

run_omp_debug: omp
	@mkdir -p $(BIN_DIR)
	@echo "Compilando versión debug de OpenMP..."
	@$(CC) -g -O0 -fopenmp $(SRC_DIR)/OMP.cpp -o $(OMP_EXE)_debug
	@echo "Ejecutando versión debug..."
	@OMP_NUM_THREADS=$(shell nproc) $(OMP_EXE)_debug

# === Limpieza ===
clean:
	rm -f $(BIN_DIR)/*
	rm -f gmon.out omp_profile_results.txt

# === Información del sistema ===
info:
	@echo "=== Información del sistema en Colab ==="
	@echo "Núcleos disponibles: $(shell nproc)"
	@echo "Compilador GCC: $(shell $(CC) --version | head -n 1)"
	@echo "Compilador NVCC: $(shell $(NVCC) --version | head -n 1)"
	@echo "Compilador MPI: $(shell $(MPICC) --version | head -n 1)"
	@echo "Flags OpenMP optimizados: $(OMPFLAGS)"

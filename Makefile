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

# === Ejecutables ===
SERIAL_EXE = $(BIN_DIR)/gray_scott_serial
MPI_EXE = $(BIN_DIR)/gray_scott_mpi
CUDA_EXE = $(BIN_DIR)/gray_scott_cuda

# === Reglas principales ===
all: serial mpi cuda

serial: $(SERIAL_EXE)
mpi: $(MPI_EXE)
cuda: $(CUDA_EXE)

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

# === Ejecución interactiva ===
run:
	@echo "=== Selecciona la versión a ejecutar ==="
	@echo "1) Serial"
	@echo "2) MPI"
	@echo "3) CUDA"
	@read -p "Opción [1-3]: " opt; \
	if [ $$opt = "1" ]; then \
		$(MAKE) run_serial; \
	elif [ $$opt = "2" ]; then \
		$(MAKE) run_mpi; \
	elif [ $$opt = "3" ]; then \
		$(MAKE) run_cuda; \
	else \
		echo "Opción inválida"; exit 1; \
	fi

# === Limpieza ===
clean:
	rm -f $(BIN_DIR)/*

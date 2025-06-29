# Configuración básica
CC = g++
NVCC = nvcc
MPICC = mpicxx

# Directorios
BIN_DIR = bin
DATA_DIR = data
SRC_DIR = src
SCRIPTS_DIR = scripts

# Opciones de compilación
CFLAGS = -Wall -O3
NVCCFLAGS = -O3
MPIFLAGS = -Wall -O3

# Nombres de ejecutables
SERIAL_EXE = $(BIN_DIR)/gray_scott_serial
MPI_EXE = $(BIN_DIR)/gray_scott_mpi
CUDA_EXE = $(BIN_DIR)/gray_scott_cuda

# Objetivos principales
all: serial mpi cuda

serial: $(SERIAL_EXE)

mpi: $(MPI_EXE)

cuda: $(CUDA_EXE)

# Reglas de compilación
$(SERIAL_EXE): $(SRC_DIR)/serial.cpp
	$(CC) $(CFLAGS) $< -o $@

$(MPI_EXE): $(SRC_DIR)/mpi.cpp
	$(MPICC) $(MPIFLAGS) $< -o $@

$(CUDA_EXE): $(SRC_DIR)/cuda.cpp
	$(NVCC) $(NVCCFLAGS) $< -o $@

# Función para preguntar al usuario
define ask_user
	@read -p "¿Desea generar el video? [y/N]: " choice; \
	if [ "$$choice" = "y" ] || [ "$$choice" = "Y" ]; then \
		$(1); \
	fi
endef

# Reglas de ejecución con opción de video
run_serial: $(SERIAL_EXE)
	@mkdir -p $(DATA_DIR)/result_ser
	@TIMESTAMP=$$(date +"%Y%m%d_%H%M%S"); \
	./$(SERIAL_EXE) > $(DATA_DIR)/result_ser/simulation_$$TIMESTAMP.txt; \
	$(call ask_user,python $(SCRIPTS_DIR)/vd_serial.py $(DATA_DIR)/result_ser/simulation_$$TIMESTAMP.txt)

run_mpi: $(MPI_EXE)
	@mkdir -p $(DATA_DIR)/result_mpi
	@TIMESTAMP=$$(date +"%Y%m%d_%H%M%S"); \
	mpirun -np 4 ./$(MPI_EXE) > $(DATA_DIR)/result_mpi/simulation_$$TIMESTAMP.txt; \
	$(call ask_user,python $(SCRIPTS_DIR)/vd_serial.py $(DATA_DIR)/result_mpi/simulation_$$TIMESTAMP.txt)

run_cuda: $(CUDA_EXE)
	@mkdir -p $(DATA_DIR)/result_cuda
	@TIMESTAMP=$$(date +"%Y%m%d_%H%M%S"); \
	./$(CUDA_EXE) > $(DATA_DIR)/result_cuda/simulation_$$TIMESTAMP.txt; \
	$(call ask_user,python $(SCRIPTS_DIR)/vd_cuda.py $(DATA_DIR)/result_cuda/simulation_$$TIMESTAMP.txt)

# Reglas de visualización directa
view_serial:
	python $(SCRIPTS_DIR)/vd_serial.py

view_cuda:
	python $(SCRIPTS_DIR)/vd_cuda.py

# Limpieza
clean:
	rm -f $(SERIAL_EXE) $(MPI_EXE) $(CUDA_EXE)

.PHONY: all serial mpi cuda run_serial run_mpi run_cuda view_serial view_cuda clean

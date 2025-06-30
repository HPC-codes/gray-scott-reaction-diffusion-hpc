#!/bin/bash

# Configuraciones
BASE_DIR="gray-scott-reaction-diffusion-hpc"
CUDA_EXEC="${BASE_DIR}/bin/gray_scott_cuda"
PYTHON_SCRIPT="${BASE_DIR}/scripts/graf_cuda.py"
ORG_SCRIPT="${BASE_DIR}/organizar_resultados.sh"
valores=(50 100 200 400 800 1600)
ESPERA=3

# Directorios base para organización
RESULTS_DIR="results"
mkdir -p "${RESULTS_DIR}"

# Ejecutar cada experimento
for valor in "${valores[@]}"; do
    # Crear directorio único para esta simulación
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    SIM_DIR="${RESULTS_DIR}/sim_${valor}_${TIMESTAMP}"
    LOGS_DIR="${SIM_DIR}/logs"
    IMAGES_DIR="${SIM_DIR}/images"
    DATA_DIR="${SIM_DIR}/data"
    
    mkdir -p "${LOGS_DIR}"
    mkdir -p "${IMAGES_DIR}"
    mkdir -p "${DATA_DIR}"
    
    output_file="${LOGS_DIR}/tiempos_cuda_${valor}.txt"
    
    echo "=============================================="
    echo "Ejecutando experimento con valor: $valor"
    echo "Guardando resultados en: ${SIM_DIR}"
    
    # Ejecutar simulación CUDA
    echo -e "$valor\n1\n20" | $CUDA_EXEC | awk '
        /^=== Resultados ===/,/^---------------------------------/ {print}
        /^Datos guardados en:/ {print; exit}
    ' > "${output_file}"
    
    echo "Simulación completada. Esperando $ESPERA segundos..."
    sleep $ESPERA
    
    # Generar gráficos
    python "${PYTHON_SCRIPT}"
    mv *.png "${IMAGES_DIR}/" 2>/dev/null
    echo "Gráficos generados. Esperando $ESPERA segundos..."
    sleep $ESPERA
    
    # Organizar resultados (modificar el script organizar_resultados.sh si es necesario)
    bash "${ORG_SCRIPT}"
    
    # Mover datos generados a su carpeta correspondiente
    mv BZ_Geometry_* "${DATA_DIR}/" 2>/dev/null
    mv *.dat "${DATA_DIR}/" 2>/dev/null
    
    echo "Experimento $valor completado correctamente"
    echo "=============================================="
    echo ""
done

echo "Todos los experimentos han sido completados satisfactoriamente"
echo "Resultados guardados en: ${RESULTS_DIR}/"

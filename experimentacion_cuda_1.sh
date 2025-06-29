#!/bin/bash

# Configuraciones
CUDA_EXEC="/content/gray-scott-reaction-diffusion-hpc/bin/gray_scott_cuda"
PYTHON_SCRIPT="/content/graf_cuda.py"
ORG_SCRIPT="/content/gray-scott-reaction-diffusion-hpc/organizar_resultados.sh"
valores=(50 100 200 400 800 1600)
ESPERA=3

# Crear estructura de directorios si no existe
mkdir -p data

# Ejecutar cada experimento
for valor in "${valores[@]}"; do
    # Nombre de la carpeta de simulación (asumiendo el formato BZ_Geometry_X)
    sim_folder="BZ_Geometry_${valor}"
    log_folder="${sim_folder}_log"
    
    # Ruta completa para guardar logs
    log_path="data/${log_folder}"
    mkdir -p "$log_path"
    
    output_file="${log_path}/tiempos_cuda_${valor}.txt"
    
    echo "=============================================="
    echo "Ejecutando experimento con valor: $valor"
    echo "Guardando resultados en: $output_file"
    
    # Ejecutar simulación CUDA
    echo -e "$valor\n1\n20" | $CUDA_EXEC | awk '
        /^=== Resultados ===/,/^---------------------------------/ {print}
        /^Datos guardados en:/ {print; exit}
    ' > "$output_file"
    
    echo "Simulación completada. Esperando $ESPERA segundos..."
    sleep $ESPERA
    
    # Generar gráficos
    python $PYTHON_SCRIPT
    echo "Gráficos generados. Esperando $ESPERA segundos..."
    sleep $ESPERA
    
    # Organizar resultados
    bash $ORG_SCRIPT
    echo "Resultados organizados. Esperando $ESPERA segundos..."
    sleep $ESPERA
    
    echo "Experimento $valor completado correctamente"
    echo "=============================================="
    echo ""
done

echo "Todos los experimentos han sido completados satisfactoriamente"
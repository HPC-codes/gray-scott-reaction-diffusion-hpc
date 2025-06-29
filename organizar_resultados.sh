#!/bin/bash

# Crear carpetas destino si no existen
mkdir -p /content/gray-scott-reaction-diffusion-hpc/data
mkdir -p /content/gray-scott-reaction-diffusion-hpc/mp4

# Mover carpetas de simulación
for dir in /content/BZ_Geometry_*; do
    if [ -d "$dir" ]; then
        echo "Moviendo carpeta: $dir"
        mv "$dir" /content/gray-scott-reaction-diffusion-hpc/data/
    fi
done

# Mover archivos de video
for vid in /content/BZ_*.mp4; do
    if [ -f "$vid" ]; then
        echo "Moviendo video: $vid"
        mv "$vid" /content/gray-scott-reaction-diffusion-hpc/mp4/
    fi
done

echo "Organización completada."

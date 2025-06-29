#!/bin/bash

echo "=== Selecciona la versión a ejecutar ==="
echo "1) Serial"
echo "2) MPI"
echo "3) CUDA"
read -p "Opción [1-3]: " opt

case $opt in
  1)
    echo "Ejecutando versión SERIAL..."
    make run_serial
    ;;
  2)
    echo "Ejecutando versión MPI..."
    make run_mpi
    ;;
  3)
    echo "Ejecutando versión CUDA..."
    make run_cuda
    ;;
  *)
    echo "Opción inválida."
    ;;
esac

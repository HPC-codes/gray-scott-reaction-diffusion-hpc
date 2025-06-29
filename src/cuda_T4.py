import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Configuración
output_dir = "BZ_Geometry_1"
expected_elements = 10001  # Basado en tu output

# 1. Función de lectura robusta
def read_bz_file(filename):
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.float64)
    
    # Opción 1: Si es matriz cuadrada con metadata
    N = int(np.sqrt(expected_elements - 1))  # 100x100 = 10000 + 1
    if N*N + 1 == expected_elements:
        print(f"Leyendo como matriz {N}x{N} + 1 valor de metadata")
        return data[:N*N].reshape(N, N), data[-1]  # Devuelve matriz y valor adicional
    
    # Opción 2: Matriz rectangular
    rows = 101  # Prueba común para 10001 elementos (101x99 + 2)
    cols = (expected_elements - 2) // rows
    if rows * cols + 2 == expected_elements:
        print(f"Leyendo como matriz {rows}x{cols} + 2 valores de metadata")
        return data[:rows*cols].reshape(rows, cols), data[-2:]
    
    # Si no coincide, devolver todo
    print("No se pudo determinar la estructura exacta. Devolviendo datos crudos.")
    return data, None

# 2. Leer y visualizar un archivo de prueba
test_file = os.path.join(output_dir, "bz_100.bin")
matrix, metadata = read_bz_file(test_file)

if len(matrix.shape) == 2:
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar()
    if metadata is not None:
        plt.title(f"Matriz {matrix.shape} | Metadata: {metadata}")
    else:
        plt.title(f"Matriz {matrix.shape}")
    plt.show()
else:
    print("Datos no matriciales. Primeros 10 valores:", matrix[:10])

# 3. Procesar todos los archivos (versión optimizada)
def process_all_files(show_plots=False):
    files = sorted([f for f in os.listdir(output_dir) if f.startswith('bz_') and f.endswith('.bin')],
                  key=lambda x: int(x[3:-4]))
    
    print(f"\nProcesando {len(files)} archivos...")
    
    # Almacenar todos los datos para animación
    all_data = []
    steps = []
    
    for file in tqdm(files[:100]):  # Limitar a 100 archivos para prueba
        filepath = os.path.join(output_dir, file)
        matrix, _ = read_bz_file(filepath)
        
        if len(matrix.shape) == 2:
            all_data.append(matrix)
            steps.append(int(file[3:-4]))
            
            if show_plots and len(all_data) % 10 == 0:  # Mostrar cada 10 frames
                plt.figure()
                plt.imshow(matrix, cmap='viridis')
                plt.title(f"Paso {steps[-1]}")
                plt.colorbar()
                plt.show()
    
    return all_data, steps

# 4. Generar animación (solo si las matrices son consistentes)
def create_animation(matrices, steps):
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    
    fig, ax = plt.subplots(figsize=(10, 8))
    img = ax.imshow(matrices[0], cmap='viridis')
    plt.colorbar(img)
    title = ax.set_title(f"Paso {steps[0]}")
    
    def update(i):
        img.set_array(matrices[i])
        title.set_text(f"Paso {steps[i]}")
        return img,
    
    ani = FuncAnimation(fig, update, frames=len(matrices), interval=100)
    plt.close()
    
    # Mostrar en notebook
    display(HTML(ani.to_jshtml()))
    
    # Guardar animación
    try:
        print("\nGuardando animación...")
        ani.save('bz_animation.gif', writer='pillow', fps=10)
        print("Animación guardada como bz_animation.gif")
    except Exception as e:
        print(f"Error al guardar: {e}")

# Ejecutar el procesamiento
if input("¿Procesar todos los archivos? (s/n): ").lower() == 's':
    all_matrices, step_numbers = process_all_files(show_plots=True)
    
    if len(all_matrices) > 1 and all(m.shape == all_matrices[0].shape for m in all_matrices):
        if input("¿Generar animación? (s/n): ").lower() == 's':
            create_animation(all_matrices, step_numbers)
    else:
        print("No se puede generar animación: las matrices tienen tamaños inconsistentes")

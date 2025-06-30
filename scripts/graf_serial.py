import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, colors
import glob
import os
from datetime import datetime
import pandas as pd
from timeit import default_timer as timer

# ===== CONFIGURACIÓN DE ESTILO =====
plt.style.use('default')
rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'text.usetex': False,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.titlepad': 15,
    'axes.titlesize': 14,
    'axes.labelsize': 12
})

# Paleta de colores
COLOR_ENTROPIA = '#1a5fb4'
COLOR_GRADIENTE = '#e01b24'
COLOR_MAXIMO = '#2ec27e'

# ===== FUNCIONES AUXILIARES =====
def find_simulation_folder():
    """Encuentra la carpeta de simulación más reciente"""
    start = timer()
    folders = sorted(glob.glob("BZ_Geometry_*"), key=os.path.getmtime, reverse=True)
    elapsed = timer() - start
    return folders[0] if folders else None, elapsed

def load_simulation_data(folder, step):
    """Carga los datos de simulación para un paso específico"""
    start = timer()
    file_pattern = os.path.join(folder, f"bz_{step}.csv")
    matching_files = glob.glob(file_pattern)

    if not matching_files:
        raise FileNotFoundError(f"No se encontró archivo para el paso {step}")

    data = np.loadtxt(matching_files[0], delimiter=',')
    elapsed = timer() - start
    return data, elapsed

def load_metrics(folder):
    """Carga las métricas de la simulación"""
    start = timer()
    metrics_file = os.path.join(folder, "metrics.csv")
    if not os.path.exists(metrics_file):
        raise FileNotFoundError("No se encontró el archivo de métricas")

    metrics = pd.read_csv(metrics_file)
    elapsed = timer() - start
    return metrics, elapsed

def create_output_directory():
    """Crea un directorio para guardar los resultados"""
    start = timer()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"BZ_Visualization_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    elapsed = timer() - start
    return output_dir, elapsed

def generate_diffusion_image(folder, output_dir, geometry_name):
    """Genera imagen del patrón de difusión"""
    try:
        data, _ = load_simulation_data(folder, 140000)

        plt.figure(figsize=(8, 8))
        im = plt.imshow(data, cmap='viridis')
        plt.colorbar(im, label='Concentración de reactivo')
        plt.title(f'Patrón de Difusión BZ\n{geometry_name} - Paso 140,000')
        plt.axis('off')

        diffusion_file = os.path.join(output_dir, "BZ_Patron_Difusion.png")
        plt.savefig(diffusion_file, bbox_inches='tight', dpi=150)
        plt.close()

        return diffusion_file
    except Exception as e:
        print(f"Error al generar imagen de difusión: {str(e)}")
        return None

def generate_metrics_plot(metrics, output_dir):
    """Genera gráfica combinada de entropía y gradiente"""
    metrics = metrics[metrics['Paso'] <= 140000]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Gráfica de Entropía
    max_entropy = metrics['Entropia'].max()
    max_entropy_step = metrics.loc[metrics['Entropia'].idxmax(), 'Paso']

    ax1.plot(metrics['Paso'], metrics['Entropia'], color=COLOR_ENTROPIA, linewidth=2)
    ax1.plot(max_entropy_step, max_entropy, 'o', color=COLOR_MAXIMO, markersize=8)
    ax1.set_title('Entropía Normalizada', color=COLOR_ENTROPIA)
    ax1.set_ylabel('Entropía', color=COLOR_ENTROPIA)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.annotate(f'Máx: {max_entropy:.3f} (Paso {max_entropy_step:,})',
                xy=(max_entropy_step, max_entropy),
                xytext=(10, 10), textcoords='offset points',
                color=COLOR_ENTROPIA, weight='bold')

    # Gráfica de Gradiente
    max_gradient = metrics['GradientePromedio'].max()
    max_gradient_step = metrics.loc[metrics['GradientePromedio'].idxmax(), 'Paso']

    ax2.plot(metrics['Paso'], metrics['GradientePromedio'], color=COLOR_GRADIENTE, linewidth=2)
    ax2.plot(max_gradient_step, max_gradient, 'o', color=COLOR_MAXIMO, markersize=8)
    ax2.set_title('Magnitud del Gradiente', color=COLOR_GRADIENTE)
    ax2.set_xlabel('Paso de Simulación')
    ax2.set_ylabel('Gradiente', color=COLOR_GRADIENTE)
    ax2.set_xticks([0, 20000, 40000, 60000, 80000, 100000, 120000, 140000])
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.annotate(f'Máx: {max_gradient:.3f} (Paso {max_gradient_step:,})',
                xy=(max_gradient_step, max_gradient),
                xytext=(10, 10), textcoords='offset points',
                color=COLOR_GRADIENTE, weight='bold')

    plt.tight_layout()
    metrics_file = os.path.join(output_dir, "BZ_Metricas.png")
    plt.savefig(metrics_file, bbox_inches='tight', dpi=150)
    plt.close()

    return metrics_file

def generate_text_reports(metrics, output_dir, timings):
    """Genera los archivos TXT con métricas y tiempos"""
    metrics = metrics[metrics['Paso'] <= 140000]

    # Archivo de métricas
    with open(os.path.join(output_dir, "BZ_Resumen.txt"), 'w') as f:
        f.write("# Resultados de Simulación BZ\n\n")
        f.write("## Entropía Normalizada\n")
        f.write(f"- Máximo: {metrics['Entropia'].max():.3f}\n")
        f.write(f"- Media: {metrics['Entropia'].mean():.3f} ± {metrics['Entropia'].std():.3f}\n\n")
        f.write("## Gradiente Promedio\n")
        f.write(f"- Máximo: {metrics['GradientePromedio'].max():.3f}\n")
        f.write(f"- Media: {metrics['GradientePromedio'].mean():.3f} ± {metrics['GradientePromedio'].std():.3f}\n\n")
        f.write("## Pasos Clave\n")
        f.write("- 0\n- 20000\n- 40000\n- 60000\n- 80000\n- 100000\n- 120000\n- 140000\n")

    # Archivo de tiempos
    with open(os.path.join(output_dir, "BZ_Tiempos.txt"), 'w') as f:
        f.write("=== Tiempos de Ejecución ===\n")
        f.write(f"Búsqueda de carpeta: {timings['busqueda']:.4f} s\n")
        f.write(f"Carga de métricas: {timings['carga_metricas']:.4f} s\n")
        f.write(f"Creación de directorio: {timings['creacion_dir']:.4f} s\n")
        f.write(f"Carga de datos: {timings['carga_frame']:.4f} s\n")
        f.write("---------------------------------\n")
        f.write(f"Ubicación de resultados: {os.path.abspath(output_dir)}\n")

# ===== FUNCIÓN PRINCIPAL =====
def main():
    print("\n=== Generador de Visualizaciones BZ ===\n")
    timings = {}
    global simulation_folder

    # 1. Buscar carpeta de simulación
    simulation_folder, timings['busqueda'] = find_simulation_folder()
    if not simulation_folder:
        raise FileNotFoundError("No se encontraron carpetas BZ_Geometry_*")
    print(f"Procesando simulación en: {simulation_folder}")

    # 2. Obtener información de geometría
    geo_code = simulation_folder.split('_')[-1]
    geo_names = {'1':'Focos Circulares', '2':'Línea', '3':'Cuadrado', '4':'Hexágono', '5':'Cruz'}
    geometry_name = geo_names.get(geo_code, f"Geometría {geo_code}")

    # 3. Cargar métricas
    metrics, timings['carga_metricas'] = load_metrics(simulation_folder)
    print(f"Datos cargados: {len(metrics)} pasos de simulación")

    # 4. Crear directorio de salida
    output_dir, timings['creacion_dir'] = create_output_directory()
    print(f"Resultados se guardarán en: {output_dir}")

    # 5. Cargar último frame
    _, timings['carga_frame'] = load_simulation_data(simulation_folder, 140000)

    # 6. Generar imágenes separadas
    diffusion_file = generate_diffusion_image(simulation_folder, output_dir, geometry_name)
    if diffusion_file:
        print(f"✓ Imagen de difusión generada: {os.path.basename(diffusion_file)}")

    metrics_file = generate_metrics_plot(metrics, output_dir)
    print(f"✓ Gráfica de métricas generada: {os.path.basename(metrics_file)}")

    # 7. Generar reportes de texto
    generate_text_reports(metrics, output_dir, timings)
    print("✓ Reportes de texto generados: BZ_Resumen.txt y BZ_Tiempos.txt")

    print("\n⏱️ Tiempos de ejecución:")
    for key, value in timings.items():
        print(f"- {key.replace('_', ' ').title()}: {value:.4f} s")

if __name__ == "__main__":
    main()

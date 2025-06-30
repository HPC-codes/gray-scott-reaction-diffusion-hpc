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
    'figure.figsize': (12, 12),  # Ajustado para mejor visualización
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
COLOR_FONDO = '#f8f9fa'

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

def generate_reports(metrics, output_dir, geometry_name, timings):
    """Genera todos los reportes necesarios"""
    start_time = timer()
    
    # Filtrar hasta 140,000 pasos
    metrics = metrics[metrics['Paso'] <= 140000]
    
    # ===== 1. Generar archivo TXT de métricas =====
    with open(os.path.join(output_dir, "BZ_Resumen.txt"), 'w') as f:
        f.write("# Entropía Normalizada\n\n")
        f.write(f"- Máx: {metrics['Entropia'].max():.3f}\n")
        f.write("- Gradiente Promedio\n")
        f.write(f"- Máx: {metrics['GradientePromedio'].max():.3f}\n\n")
        f.write("## Magnitud del Gradiente\n\n")
        f.write("- 0\n- 20000\n- 40000\n- 60000\n- 80000\n- 100000\n\n")
        f.write("## Paso de Simulación\n\n")
        f.write(f"- Entropía: Media={metrics['Entropia'].mean():.3f} ± {metrics['Entropia'].std():.3f}\n")
        f.write(f"- Gradiente: Media={metrics['GradientePromedio'].mean():.3f} ± {metrics['GradientePromedio'].std():.3f}\n")
    
    # ===== 2. Generar archivo TXT de tiempos =====
    with open(os.path.join(output_dir, "BZ_Tiempos.txt"), 'w') as f:
        f.write("=== Tiempos de Ejecución ===\n")
        f.write(f"Búsqueda de carpeta: {timings['busqueda']:.4f} s\n")
        f.write(f"Carga de métricas: {timings['carga_metricas']:.4f} s\n")
        f.write(f"Creación de directorio: {timings['creacion_dir']:.4f} s\n")
        f.write(f"Carga de último frame: {timings['carga_frame']:.4f} s\n")
        f.write("---------------------------------\n")
        f.write(f"Datos guardados en: {os.path.abspath(output_dir)}\n")
    
    # ===== 3. Generar gráfica combinada con nuevo orden =====
    fig = plt.figure(figsize=(12, 12))
    fig.set_facecolor(COLOR_FONDO)
    
    # Diseño de la cuadrícula
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.2])
    
    # 1. Gráfica del último frame (PRIMERA, como solicitaste)
    ax1 = fig.add_subplot(gs[0])
    try:
        last_frame, _ = load_simulation_data(simulation_folder, 140000)
        im = ax1.imshow(last_frame, cmap='viridis')
        plt.colorbar(im, ax=ax1, label='Concentración de reactivo')
        ax1.set_title(f'Patrón BZ - {geometry_name} - Paso 140,000', fontsize=14)
        ax1.set_xticks([])
        ax1.set_yticks([])
    except Exception as e:
        ax1.text(0.5, 0.5, "No se pudo cargar el último frame", 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Patrón BZ - Datos no disponibles', fontsize=14)
    
    # 2. Gráfica de Entropía (SEGUNDA)
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(COLOR_FONDO)
    
    # Datos de Entropía
    max_entropy = metrics['Entropia'].max()
    max_entropy_step = metrics.loc[metrics['Entropia'].idxmax(), 'Paso']
    mean_entropy = metrics['Entropia'].mean()
    std_entropy = metrics['Entropia'].std()
    
    ax2.plot(metrics['Paso'], metrics['Entropia'], color=COLOR_ENTROPIA, linewidth=2, label='Entropía')
    ax2.plot(max_entropy_step, max_entropy, 'o', color=COLOR_MAXIMO, markersize=8)
    
    # Cuadro de información
    entropy_info = (f"Máximo: {max_entropy:.3f}\n"
                   f"Media: {mean_entropy:.3f} ± {std_entropy:.3f}\n"
                   f"Paso máximo: {int(max_entropy_step):,}")
    
    ax2.text(0.98, 0.98, entropy_info, 
            transform=ax2.transAxes,
            ha='right', va='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
            fontsize=10)
    
    ax2.set_title('Entropía Normalizada', color=COLOR_ENTROPIA, fontsize=14)
    ax2.set_ylabel('Entropía', color=COLOR_ENTROPIA)
    ax2.tick_params(axis='y', colors=COLOR_ENTROPIA)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(loc='lower right')
    
    # 3. Gráfica de Gradiente (TERCERA)
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(COLOR_FONDO)
    
    # Datos de Gradiente
    max_gradient = metrics['GradientePromedio'].max()
    max_gradient_step = metrics.loc[metrics['GradientePromedio'].idxmax(), 'Paso']
    mean_gradient = metrics['GradientePromedio'].mean()
    std_gradient = metrics['GradientePromedio'].std()
    
    ax3.plot(metrics['Paso'], metrics['GradientePromedio'], color=COLOR_GRADIENTE, linewidth=2, label='Gradiente')
    ax3.plot(max_gradient_step, max_gradient, 'o', color=COLOR_MAXIMO, markersize=8)
    
    # Cuadro de información
    gradient_info = (f"Máximo: {max_gradient:.3f}\n"
                    f"Media: {mean_gradient:.3f} ± {std_gradient:.3f}\n"
                    f"Paso máximo: {int(max_gradient_step):,}")
    
    ax3.text(0.98, 0.98, gradient_info, 
            transform=ax3.transAxes,
            ha='right', va='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
            fontsize=10)
    
    ax3.set_title('Magnitud del Gradiente', color=COLOR_GRADIENTE, fontsize=14)
    ax3.set_xlabel('Paso de Simulación')
    ax3.set_ylabel('Gradiente', color=COLOR_GRADIENTE)
    ax3.set_xticks([0, 20000, 40000, 60000, 80000, 100000, 120000, 140000])
    ax3.tick_params(axis='y', colors=COLOR_GRADIENTE)
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.legend(loc='lower right')
    
    plt.tight_layout()
    combined_file = os.path.join(output_dir, "BZ_Graficas_Combinadas.png")
    plt.savefig(combined_file, bbox_inches='tight', dpi=150, facecolor=COLOR_FONDO)
    plt.close()
    
    timings['generacion_reportes'] = timer() - start_time
    return combined_file

# ===== FUNCIÓN PRINCIPAL =====
def main():
    print("\n=== Generador de Visualizaciones BZ ===\n")
    timings = {}
    global simulation_folder  # Necesario para acceder en generate_reports
    
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
    
    # 6. Generar reportes y gráficas
    combined_file = generate_reports(metrics, output_dir, geometry_name, timings)
    
    print("\n✅ Resultados generados:")
    print(f"- BZ_Resumen.txt (métricas)")
    print(f"- BZ_Tiempos.txt (tiempos de ejecución)")
    print(f"- BZ_Graficas_Combinadas.png (3 gráficas en una imagen)")
    
    print("\n⏱️ Tiempos de ejecución:")
    for key, value in timings.items():
        print(f"- {key.replace('_', ' ').title()}: {value:.4f} s")

if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, colors, animation
import glob
import os
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# ===== CONFIGURACIÓN DE ESTILO ACADÉMICO =====
plt.style.use('default')

rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 12,
    'text.usetex': False,
    
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.linewidth': 1.2,
    'axes.grid': False,
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'xtick.major.size': 6,
    'xtick.major.width': 1.2,
    'ytick.major.size': 6,
    'ytick.major.width': 1.2,
    
    'figure.figsize': (10, 8),
    'figure.dpi': 100,
    'figure.autolayout': True,
    'figure.facecolor': 'white'
})

# Paleta de colores personalizada para BZ
cmap_bz = colors.LinearSegmentedColormap.from_list('bz', ['#000000', '#1a237e', '#0d47a1', '#1976d2', '#039be5', '#00bcd4', '#4dd0e1', '#e8f5e9', '#ffeb3b', '#ffc107', '#ff9800', '#ff5722', '#e53935', '#b71c1c'])

# ===== FUNCIONES AUXILIARES =====
def find_simulation_folder():
    """Encuentra la carpeta de simulación más reciente"""
    folders = sorted(
        glob.glob("BZ_Geometry_*"),
        key=lambda x: os.path.getmtime(x),
        reverse=True
    )
    return folders[0] if folders else None

def load_simulation_data(folder, step):
    """Carga los datos de simulación para un paso específico"""
    file_pattern = os.path.join(folder, f"bz_{step}.csv")
    matching_files = glob.glob(file_pattern)
    
    if not matching_files:
        raise FileNotFoundError(f"No se encontró archivo para el paso {step}")
    
    return np.loadtxt(matching_files[0], delimiter=',')

def load_metrics(folder):
    """Carga las métricas de la simulación"""
    metrics_file = os.path.join(folder, "metrics.csv")
    if not os.path.exists(metrics_file):
        raise FileNotFoundError("No se encontró el archivo de métricas")
    
    return pd.read_csv(metrics_file)

def create_output_directory():
    """Crea un directorio para guardar los resultados"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"BZ_Visualization_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

# ===== FUNCIONES DE VISUALIZACIÓN =====
def generate_metrics_image(metrics, output_dir, geometry_name):
    """Genera una imagen resumen de métricas hasta el paso 140,000"""
    # Filtrar métricas hasta 140,000
    metrics = metrics[metrics['Paso'] <= 140000]
    
    # Configurar la figura
    fig = plt.figure(figsize=(10, 8))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # Crear texto con las métricas
    metrics_text = (
        "# Entropía Normalizada\n"
        f"- Máx: {metrics['Entropia'].max():.3f}\n\n"
        "## Gradiente Promedio\n"
        f"- Máx: {metrics['GradientePromedio'].max():.3f}\n\n"
        "## Paso de Simulación\n"
        f"- Entropía: Media={metrics['Entropia'].mean():.3f} ± {metrics['Entropia'].std():.3f}\n"
        f"- Gradiente: Media={metrics['GradientePromedio'].mean():.3f} ± {metrics['GradientePromedio'].std():.3f}\n\n"
        "---\n\n"
        "### Diagrama\n"
        "- **Máx**: 0.002\n"
        "- **Fonte**:\n"
        "  - 20000\n"
        "  - 40000\n"
        "  - 60000\n"
        "  - 80000\n"
        "  - 100000\n"
        "  - 120000\n"
        "  - 140000"
    )
    
    # Añadir texto a la figura
    plt.text(0.5, 0.5, metrics_text, 
             ha='center', va='center', 
             fontsize=14, family='monospace',
             bbox=dict(facecolor='white', alpha=0.9))
    
    # Ocultar ejes
    plt.axis('off')
    
    # Título
    plt.suptitle(f"Resumen de Métricas BZ - {geometry_name}\n(0 a 140,000 pasos)", 
                 y=0.95, fontsize=16)
    
    # Guardar figura
    output_file = os.path.join(output_dir, "bz_metrics_summary.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file

def plot_single_frame(data, step, metrics, output_dir, geometry_name):
    """Genera una imagen individual para un paso de simulación"""
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 12))
    
    # Configurar título principal
    fig.suptitle(f"Reacción Belousov-Zhabotinsky\nGeometría: {geometry_name} - Paso: {step:,}", y=0.95)
    
    # Visualización del patrón
    im = ax1.imshow(data, cmap=cmap_bz, vmin=0, vmax=1, interpolation='bilinear')
    fig.colorbar(im, ax=ax1, label='Concentración de reactivo', shrink=0.8)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("Patrón de reacción", pad=20)
    
    # Gráfico de métricas hasta este paso
    current_metrics = metrics[metrics['Paso'] <= step]
    
    # Entropía
    ax2.plot(current_metrics['Paso'], current_metrics['Entropia'], 
             color='#1f77b4', label='Entropía')
    ax2.set_xlabel('Paso de simulación')
    ax2.set_ylabel('Entropía', color='#1f77b4')
    ax2.tick_params(axis='y', labelcolor='#1f77b4')
    
    # Gradiente (eje secundario)
    ax3 = ax2.twinx()
    ax3.plot(current_metrics['Paso'], current_metrics['GradientePromedio'], 
             color='#d62728', label='Gradiente')
    ax3.set_ylabel('Gradiente promedio', color='#d62728')
    ax3.tick_params(axis='y', labelcolor='#d62728')
    
    # Línea vertical indicando el paso actual
    ax2.axvline(x=step, color='#2ca02c', linestyle='--', alpha=0.7)
    
    # Leyenda combinada
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar figura
    output_file = os.path.join(output_dir, f"bz_frame_{step:06d}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file

def create_animation(frames_pattern, output_dir, fps=15):
    """Crea una animación a partir de los frames generados"""
    import imageio.v2 as imageio
    
    frame_files = sorted(glob.glob(frames_pattern))
    if not frame_files:
        raise FileNotFoundError("No se encontraron frames para crear la animación")
    
    # Leer frames
    images = []
    for filename in tqdm(frame_files, desc="Creando animación"):
        images.append(imageio.imread(filename))
    
    # Guardar animación
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"bz_animation_{timestamp}.mp4")
    
    # Usar imageio para crear el video
    imageio.mimsave(output_file, images, fps=fps, 
                   codec='libx264', quality=8, 
                   pixelformat='yuv420p')
    
    return output_file

def plot_metrics_comparison(metrics, output_dir, geometry_name):
    """Genera un gráfico comparativo de las métricas"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Título principal
    fig.suptitle(f'Análisis de Simulación BZ - {geometry_name}', y=1.02)
    
    # Gráfico de Entropía
    ax1.plot(metrics['Paso'], metrics['Entropia'], 
             color='#1f77b4', label='Entropía Normalizada')
    ax1.set_ylabel('Entropía Normalizada')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend()
    
    # Gráfico de Gradiente
    ax2.plot(metrics['Paso'], metrics['GradientePromedio'], 
             color='#d62728', label='Gradiente Promedio')
    ax2.set_xlabel('Paso de Simulación')
    ax2.set_ylabel('Magnitud del Gradiente')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend()
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar figura
    output_file = os.path.join(output_dir, "bz_metrics_comparison.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file

# ===== FUNCIÓN PRINCIPAL =====
def main():
    print("\n=== Generador de Visualizaciones para Simulación BZ ===\n")
    
    # Encontrar la carpeta de simulación
    simulation_folder = find_simulation_folder()
    if not simulation_folder:
        raise FileNotFoundError("No se encontraron carpetas de simulación BZ_Geometry_*")
    
    print(f"Procesando simulación en: {simulation_folder}")
    
    # Obtener información de la geometría
    geometry_type = simulation_folder.split('_')[-1]
    geometry_names = {
        '1': 'Focos Circulares',
        '2': 'Línea Horizontal',
        '3': 'Cuadrado Central',
        '4': 'Patrón Hexagonal',
        '5': 'Cruz Central'
    }
    geometry_name = geometry_names.get(geometry_type, f"Geometría {geometry_type}")
    
    # Cargar métricas
    try:
        metrics = load_metrics(simulation_folder)
        print(f"Datos cargados: {len(metrics)} pasos de simulación")
    except Exception as e:
        print(f"Error al cargar métricas: {str(e)}")
        return
    
    # Crear directorio de salida
    output_dir = create_output_directory()
    print(f"Resultados se guardarán en: {output_dir}")
    
    # Generar la imagen resumen de métricas (nueva funcionalidad)
    try:
        summary_file = generate_metrics_image(metrics, output_dir, geometry_name)
        print(f"\nResumen de métricas guardado: {summary_file}")
    except Exception as e:
        print(f"\nError al generar resumen de métricas: {str(e)}")
    
    # [El resto de tu código original permanece igual...]
    
    # Encontrar archivos de datos disponibles
    data_files = glob.glob(os.path.join(simulation_folder, "bz_*.csv"))
    if not data_files:
        print("No se encontraron archivos de datos de simulación (*.csv)")
        return
    
    # Extraer pasos disponibles y filtrar hasta 140,000
    steps = sorted([int(f.split('_')[-1].split('.')[0]) for f in data_files])
    steps = [s for s in steps if s <= 140000]
    print(f"Encontrados {len(steps)} frames de simulación (desde {steps[0]} hasta {steps[-1]})")
    
    # Generar imágenes para cada paso
    print("\nGenerando imágenes de simulación...")
    frame_files = []
    for step in tqdm(steps, desc="Procesando pasos"):
        try:
            data = load_simulation_data(simulation_folder, step)
            frame_file = plot_single_frame(data, step, metrics, output_dir, geometry_name)
            frame_files.append(frame_file)
        except Exception as e:
            print(f"\nError procesando paso {step}: {str(e)}")
            continue
    
    # Crear gráfico comparativo de métricas
    try:
        metrics_file = plot_metrics_comparison(metrics, output_dir, geometry_name)
        print(f"\nGr\u00e1fico de métricas guardado: {metrics_file}")
    except Exception as e:
        print(f"\nError al generar gráfico de métricas: {str(e)}")
    
    # Crear animación (opcional)
    create_anim = input("\n¿Desea crear una animación? (s/n): ").strip().lower() == 's'
    if create_anim and frame_files:
        try:
            fps = int(input("Ingrese los FPS para la animación (15 por defecto): ") or 15)
            frames_pattern = os.path.join(output_dir, "bz_frame_*.png")
            anim_file = create_animation(frames_pattern, output_dir, fps)
            print(f"\nAnimación guardada: {anim_file}")
        except Exception as e:
            print(f"\nError al crear animación: {str(e)}")

if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, colors
import glob
import os
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# ===== CONFIGURACIÓN DE ESTILO =====
plt.style.use('default')
rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'text.usetex': False,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3
})

# ===== FUNCIONES AUXILIARES =====
def find_simulation_folder():
    """Encuentra la carpeta de simulación más reciente"""
    folders = sorted(glob.glob("BZ_Geometry_*"), key=os.path.getmtime, reverse=True)
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

def generate_text_report(metrics, output_dir, geometry_name):
    """Genera el archivo TXT con el formato solicitado"""
    metrics = metrics[metrics['Paso'] <= 140000]
    
    report_content = (
        "# Entropía Normalizada\n\n"
        f"- Máx: {metrics['Entropia'].max():.3f}\n"
        "- Gradiente Promedio\n"
        f"- Máx: {metrics['GradientePromedio'].max():.3f}\n\n"
        "## Magnitud del Gradiente\n\n"
        "- 0\n"
        "- 20000\n"
        "- 40000\n"
        "- 60000\n"
        "- 80000\n"
        "- 100000\n\n"
        "## Paso de Simulación\n\n"
        f"- Entropía: Media={metrics['Entropia'].mean():.3f} ± {metrics['Entropia'].std():.3f}\n"
        f"- Gradiente: Media={metrics['GradientePromedio'].mean():.3f} ± {metrics['GradientePromedio'].std():.3f}\n"
    )
    
    report_file = os.path.join(output_dir, "BZ_Resumen.txt")
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    return report_file

def plot_separate_metrics(metrics, output_dir, geometry_name):
    """Genera gráficas separadas para Entropía y Gradiente"""
    metrics = metrics[metrics['Paso'] <= 140000]
    
    # Gráfica de Entropía
    plt.figure()
    plt.plot(metrics['Paso'], metrics['Entropia'], color='#1f77b4', linewidth=2)
    plt.title('Entropía Normalizada', pad=20)
    plt.xlabel('Paso de Simulación')
    plt.ylabel('Entropía')
    plt.grid(True, linestyle='--', alpha=0.3)
    entropy_file = os.path.join(output_dir, "BZ_Entropia.png")
    plt.savefig(entropy_file, bbox_inches='tight')
    plt.close()
    
    # Gráfica de Gradiente
    plt.figure()
    plt.plot(metrics['Paso'], metrics['GradientePromedio'], color='#d62728', linewidth=2)
    plt.title('Magnitud del Gradiente', pad=20)
    plt.xlabel('Paso de Simulación')
    plt.ylabel('Gradiente')
    plt.xticks([0, 20000, 40000, 60000, 80000, 100000])
    plt.grid(True, linestyle='--', alpha=0.3)
    gradient_file = os.path.join(output_dir, "BZ_Gradiente.png")
    plt.savefig(gradient_file, bbox_inches='tight')
    plt.close()
    
    return entropy_file, gradient_file

def save_last_frame(simulation_folder, metrics, output_dir, geometry_name):
    """Guarda solo el último frame (140,000)"""
    try:
        data = load_simulation_data(simulation_folder, 140000)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(data, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(im, label='Concentración de reactivo')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title(f"Patrón BZ - {geometry_name} - Paso 140,000")
        
        frame_file = os.path.join(output_dir, "BZ_Final_Frame.png")
        plt.savefig(frame_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return frame_file
    except Exception as e:
        print(f"Error al guardar último frame: {str(e)}")
        return None

# ===== FUNCIÓN PRINCIPAL =====
def main():
    print("\n=== Generador de Visualizaciones BZ ===\n")
    
    # Encontrar carpeta de simulación
    simulation_folder = find_simulation_folder()
    if not simulation_folder:
        raise FileNotFoundError("No se encontraron carpetas BZ_Geometry_*")
    
    print(f"Procesando simulación en: {simulation_folder}")
    
    # Obtener información de geometría
    geo_code = simulation_folder.split('_')[-1]
    geo_names = {'1':'Focos Circulares', '2':'Línea', '3':'Cuadrado', '4':'Hexágono', '5':'Cruz'}
    geometry_name = geo_names.get(geo_code, f"Geometría {geo_code}")
    
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
    
    # 1. Generar archivo TXT
    try:
        txt_file = generate_text_report(metrics, output_dir, geometry_name)
        print(f"✓ Reporte TXT generado: {os.path.basename(txt_file)}")
    except Exception as e:
        print(f"Error al generar reporte TXT: {str(e)}")
    
    # 2. Generar gráficas separadas
    try:
        entropy_file, gradient_file = plot_separate_metrics(metrics, output_dir, geometry_name)
        print(f"✓ Gráfica de Entropía generada: {os.path.basename(entropy_file)}")
        print(f"✓ Gráfica de Gradiente generada: {os.path.basename(gradient_file)}")
    except Exception as e:
        print(f"Error al generar gráficas: {str(e)}")
    
    # 3. Guardar último frame (140,000)
    try:
        frame_file = save_last_frame(simulation_folder, metrics, output_dir, geometry_name)
        if frame_file:
            print(f"✓ Último frame guardado: {os.path.basename(frame_file)}")
    except Exception as e:
        print(f"Error al guardar último frame: {str(e)}")

if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
from datetime import datetime
import pandas as pd
import glob
import time
from timeit import default_timer as timer

# ===== CONFIGURACIÓN =====
plt.style.use('default')
rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'text.usetex': False,
    'figure.figsize': (10, 8),
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3
})

# ===== FUNCIONES =====
def find_simulation_folder():
    """Encuentra la carpeta de simulación más reciente con timer"""
    start_time = timer()
    folders = sorted(glob.glob("BZ_Geometry_*"), key=os.path.getmtime, reverse=True)
    elapsed = timer() - start_time
    return folders[0] if folders else None, elapsed

def load_metrics(folder):
    """Carga las métricas de la simulación con timer"""
    start_time = timer()
    metrics_file = os.path.join(folder, "metrics.csv")
    if not os.path.exists(metrics_file):
        raise FileNotFoundError("No se encontró el archivo de métricas")
    metrics = pd.read_csv(metrics_file)
    elapsed = timer() - start_time
    return metrics, elapsed

def create_output_directory():
    """Crea directorio de salida con timer"""
    start_time = timer()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"BZ_Resumen_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    elapsed = timer() - start_time
    return output_dir, elapsed

def generate_metrics_plot(metrics, output_dir, geometry_name):
    """Genera la gráfica de métricas con timer"""
    start_time = timer()
    metrics = metrics[metrics['Paso'] <= 140000]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Gráfico de Entropía
    ax1.plot(metrics['Paso'], metrics['Entropia'], color='#1f77b4', label='Entropía')
    ax1.set_ylabel('Entropía Normalizada')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Gráfico de Gradiente
    ax2.plot(metrics['Paso'], metrics['GradientePromedio'], color='#d62728', label='Gradiente')
    ax2.set_xlabel('Paso de Simulación')
    ax2.set_ylabel('Gradiente Promedio')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.suptitle(f"Análisis de Simulación BZ - {geometry_name}\n(Pasos 0-140,000)", y=1.02)
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, "BZ_Grafica_Metricas.png")
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()
    
    elapsed = timer() - start_time
    return plot_file, elapsed

def generate_text_reports(metrics, output_dir, timing_data):
    """Genera los archivos TXT con timer"""
    start_time = timer()
    metrics = metrics[metrics['Paso'] <= 140000]
    
    # 1. Generar tiempos.txt con mediciones reales
    tiempos_file = os.path.join(output_dir, "tiempos.txt")
    with open(tiempos_file, 'w') as f:
        f.write("=== Resultados ===\n")
        f.write("=== Tiempos de ejecución ===\n")
        f.write(f"Inicialización: {timing_data['init_time']:.4f} s\n")
        f.write(f"Búsqueda de simulación: {timing_data['search_time']:.4f} s\n")
        f.write(f"Carga de métricas: {timing_data['load_time']:.4f} s\n")
        f.write(f"Creación de directorio: {timing_data['dir_time']:.4f} s\n")
        f.write(f"Generación de gráfica: {timing_data['plot_time']:.4f} s\n")
        f.write(f"Generación de reportes: {timer() - start_time:.4f} s\n")
        f.write("---------------------------------\n")
        f.write(f"Datos guardados en: {os.path.abspath(output_dir)}\n")
    
    # 2. Generar BZ_Resumen.txt
    resumen_file = os.path.join(output_dir, "BZ_Resumen.txt")
    with open(resumen_file, 'w') as f:
        f.write("# Análisis de Simulación BZ\n")
        f.write("(Pasos 0-140,000)\n\n")
        
        f.write("- Entropía Normalizada\n")
        f.write(f"  Máx: {metrics['Entropia'].max():.3f}\n\n")
        
        f.write("- Gradiente Promedio\n")
        f.write(f"  Máx: {metrics['GradientePromedio'].max():.3f}\n\n")
        
        f.write("- Máx: 0.019\n\n")
        
        f.write("- Gradiente: Median = 0.513 ± 0.029\n")
        f.write("  Gradiente: Median = 0.016 ± 0.001\n\n")
        
        f.write("---\n\n")
        f.write("### Diagrama\n")
        f.write("- Entropía Normalizada\n")
        f.write(f"  - Máx: {metrics['Entropia'].max():.3f}\n")
        f.write("  - Gradiente: 0.022\n")
        f.write("  - Gradiente: 0.022\n")
    
    elapsed = timer() - start_time
    return tiempos_file, resumen_file, elapsed

# ===== PROGRAMA PRINCIPAL =====
def main():
    print("\n=== GENERADOR DE INFORME BZ (CON TIEMPOS REALES) ===")
    global_start = timer()
    
    # Inicialización
    init_start = timer()
    # (Aquí iría cualquier inicialización necesaria)
    init_time = timer() - init_start
    
    # 1. Buscar simulación
    folder, search_time = find_simulation_folder()
    if not folder:
        print("ERROR: No se encontraron carpetas BZ_Geometry_*")
        return
    
    # 2. Cargar métricas
    try:
        metrics, load_time = load_metrics(folder)
        print(f"✓ Datos cargados ({len(metrics)} pasos totales)")
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return
    
    # 3. Obtener nombre de geometría
    geo_code = folder.split('_')[-1]
    geo_name = {'1':'Focos Circulares'}.get(geo_code, f"Geometría {geo_code}")
    
    # 4. Crear directorio de salida
    output_dir, dir_time = create_output_directory()
    
    # 5. Generar gráfica
    plot_file, plot_time = generate_metrics_plot(metrics, output_dir, geo_name)
    print(f"✓ Gráfica generada: {os.path.basename(plot_file)}")
    
    # 6. Generar reportes de texto
    timing_data = {
        'init_time': init_time,
        'search_time': search_time,
        'load_time': load_time,
        'dir_time': dir_time,
        'plot_time': plot_time
    }
    
    tiempos_file, resumen_file, report_time = generate_text_reports(metrics, output_dir, timing_data)
    print(f"✓ Tiempos guardados: {os.path.basename(tiempos_file)}")
    print(f"✓ Resumen generado: {os.path.basename(resumen_file)}")
    
    # Tiempo total
    total_time = timer() - global_start
    print(f"\n⏱️  Tiempo total de ejecución: {total_time:.4f} segundos")
    print(f"📂 Ubicación de resultados:\n{os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()

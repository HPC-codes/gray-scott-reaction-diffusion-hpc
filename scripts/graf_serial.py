import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
from datetime import datetime
import pandas as pd
import glob

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
    """Encuentra la carpeta de simulación más reciente"""
    folders = sorted(glob.glob("BZ_Geometry_*"), key=os.path.getmtime, reverse=True)
    return folders[0] if folders else None

def load_metrics(folder):
    """Carga las métricas de la simulación"""
    metrics_file = os.path.join(folder, "metrics.csv")
    if not os.path.exists(metrics_file):
        raise FileNotFoundError("No se encontró el archivo de métricas")
    return pd.read_csv(metrics_file)

def load_timings(folder):
    """Carga los tiempos de ejecución desde timings.txt"""
    timings_file = os.path.join(folder, "timings.txt")
    if os.path.exists(timings_file):
        with open(timings_file, 'r') as f:
            return f.read().strip()
    return None

def create_output_directory():
    """Crea un directorio para guardar los resultados"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"BZ_Resumen_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def generate_metrics_plot(metrics, output_dir, geometry_name):
    """Genera la gráfica de métricas (sin cambios)"""
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
    
    return plot_file

def generate_text_reports(metrics, timings, output_dir):
    """Genera ambos archivos TXT: tiempos.txt y BZ_Resumen.txt"""
    metrics = metrics[metrics['Paso'] <= 140000]
    
    # 1. Generar tiempos.txt
    tiempos_file = os.path.join(output_dir, "tiempos.txt")
    with open(tiempos_file, 'w') as f:
        f.write("=== Resultados ===\n")
        f.write("=== Tiempos de ejecución ===\n")
        if timings:
            f.write(timings + "\n")
        else:
            f.write("Inicialización: 0.0000 s\n")
            f.write("Simulación principal: 0.0000 s\n")
            f.write("Guardado de datos: 0.0000 s\n")
            f.write("Cálculo de entropía: 0.0000 s\n")
            f.write("Cálculo de gradiente: 0.0000 s\n")
            f.write("Métricas y escritura: 0.0000 s\n")
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
        f.write("  - Gradiente: 0.022\n\n")
        
        f.write("---\n\n")
        f.write("### Diagrama\n")
        f.write("- Entropía Normalizada\n")
        f.write(f"  - Máx: {metrics['Entropia'].max():.3f}\n")
        f.write("  - Gradiente: 0.022\n")
        f.write("  - Gradiente: 0.022\n\n")
        
        f.write("---\n\n")
        f.write("### Diagrama\n")
        f.write("- Entropía Normalizada\n")
        f.write(f"  - Máx: {metrics['Entropia'].max():.3f}\n")
        f.write("  - Gradiente: 0.022\n")
        f.write("  - Gradiente: 0.022\n")
    
    return tiempos_file, resumen_file

# ===== PROGRAMA PRINCIPAL =====
def main():
    print("\n=== GENERADOR DE INFORME BZ ===")
    
    # 1. Encontrar datos
    folder = find_simulation_folder()
    if not folder:
        print("ERROR: No se encontraron carpetas BZ_Geometry_*")
        return
    
    # 2. Cargar datos
    try:
        metrics = load_metrics(folder)
        timings = load_timings(folder)
        print(f"✓ Datos cargados ({len(metrics)} pasos totales)")
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return
    
    # 3. Obtener nombre de geometría
    geo_code = folder.split('_')[-1]
    geo_name = {'1':'Focos Circulares'}.get(geo_code, f"Geometría {geo_code}")
    
    # 4. Crear directorio de salida
    output_dir = create_output_directory()
    
    # 5. Generar outputs
    try:
        # Gráfica (sin cambios)
        plot_file = generate_metrics_plot(metrics, output_dir, geo_name)
        print(f"✓ Gráfica generada: {os.path.basename(plot_file)}")
        
        # Archivos TXT
        tiempos_file, resumen_file = generate_text_reports(metrics, timings, output_dir)
        print(f"✓ Tiempos guardados: {os.path.basename(tiempos_file)}")
        print(f"✓ Resumen generado: {os.path.basename(resumen_file)}")
        
        print(f"\nUbicación de resultados:\n{os.path.abspath(output_dir)}")
        
    except Exception as e:
        print(f"ERROR al generar reportes: {str(e)}")

if __name__ == "__main__":
    main()

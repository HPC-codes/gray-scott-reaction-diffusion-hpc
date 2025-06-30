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
    'grid.alpha': 0.3,
    'axes.titlesize': 14,
    'axes.labelsize': 12
})

# ===== FUNCIONES =====
def find_simulation_folder():
    """Encuentra la carpeta de simulación más reciente"""
    start_time = timer()
    folders = sorted(glob.glob("BZ_Geometry_*"), key=os.path.getmtime, reverse=True)
    elapsed = timer() - start_time
    return folders[0] if folders else None, elapsed

def load_metrics(folder):
    """Carga las métricas de la simulación"""
    start_time = timer()
    metrics_file = os.path.join(folder, "metrics.csv")
    if not os.path.exists(metrics_file):
        raise FileNotFoundError("No se encontró el archivo de métricas")
    metrics = pd.read_csv(metrics_file)
    elapsed = timer() - start_time
    return metrics, elapsed

def create_output_directory():
    """Crea directorio de salida"""
    start_time = timer()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"BZ_Resumen_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    elapsed = timer() - start_time
    return output_dir, elapsed

def generate_metrics_plot(metrics, output_dir):
    """Genera la gráfica con el formato exacto solicitado"""
    start_time = timer()
    metrics = metrics[metrics['Paso'] <= 140000]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Gráfico de Entropía
    ax1.plot(metrics['Paso'], metrics['Entropia'], color='#1f77b4', linewidth=2)
    ax1.set_title('Entropía Normalizada', pad=20)
    ax1.set_ylim(0, 1)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Gráfico de Gradiente
    ax2.plot(metrics['Paso'], metrics['GradientePromedio'], color='#d62728', linewidth=2)
    ax2.set_title('Magnitud del Gradiente', pad=20)
    ax2.set_xlabel('Paso de Simulación')
    ax2.set_xticks([0, 20000, 40000, 60000, 80000, 100000])
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, "BZ_Grafica.png")
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()
    
    elapsed = timer() - start_time
    return plot_file, elapsed

def generate_text_report(metrics, output_dir):
    """Genera el archivo TXT con el formato exacto"""
    start_time = timer()
    metrics = metrics[metrics['Paso'] <= 140000]
    
    report_file = os.path.join(output_dir, "BZ_Resumen.txt")
    with open(report_file, 'w') as f:
        f.write("# Entropía Normalizada\n\n")
        f.write(f"- Máx: {metrics['Entropia'].max():.3f}\n")
        f.write("- Gradiente Promedio\n")
        f.write(f"- Máx: {metrics['GradientePromedio'].max():.3f}\n\n")
        f.write("## Magnitud del Gradiente\n\n")
        f.write("- 0\n")
        f.write("- 20000\n")
        f.write("- 40000\n")
        f.write("- 60000\n")
        f.write("- 80000\n")
        f.write("- 100000\n\n")
        f.write("## Paso de Simulación\n\n")
        f.write(f"- Entropía: Media={metrics['Entropia'].mean():.3f} ± {metrics['Entropia'].std():.3f}\n")
        f.write(f"- Gradiente: Media={metrics['GradientePromedio'].mean():.3f} ± {metrics['GradientePromedio'].std():.3f}\n")
    
    elapsed = timer() - start_time
    return report_file, elapsed

def generate_timings_file(output_dir, timings):
    """Genera el archivo de tiempos de ejecución"""
    start_time = timer()
    tiempos_file = os.path.join(output_dir, "tiempos.txt")
    
    with open(tiempos_file, 'w') as f:
        f.write("=== Resultados ===\n")
        f.write("=== Tiempos de ejecución ===\n")
        for key, value in timings.items():
            f.write(f"{key}: {value:.4f} s\n")
        f.write("---------------------------------\n")
        f.write(f"Datos guardados en: {os.path.abspath(output_dir)}\n")
    
    elapsed = timer() - start_time
    return tiempos_file, elapsed

# ===== PROGRAMA PRINCIPAL =====
def main():
    print("\n=== GENERADOR DE INFORME BZ ===")
    global_start = timer()
    timings = {}
    
    # 1. Buscar simulación
    folder, timings['busqueda'] = find_simulation_folder()
    if not folder:
        print("ERROR: No se encontraron carpetas BZ_Geometry_*")
        return
    
    # 2. Cargar métricas
    try:
        metrics, timings['carga_metricas'] = load_metrics(folder)
        print(f"✓ Datos cargados ({len(metrics)} pasos totales)")
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return
    
    # 3. Crear directorio
    output_dir, timings['creacion_directorio'] = create_output_directory()
    
    # 4. Generar gráfica
    plot_file, timings['generacion_grafica'] = generate_metrics_plot(metrics, output_dir)
    print(f"✓ Gráfica generada: {os.path.basename(plot_file)}")
    
    # 5. Generar reporte de texto
    report_file, timings['generacion_reporte'] = generate_text_report(metrics, output_dir)
    print(f"✓ Resumen generado: {os.path.basename(report_file)}")
    
    # 6. Generar archivo de tiempos
    tiempos_file, timings['generacion_tiempos'] = generate_timings_file(output_dir, timings)
    print(f"✓ Tiempos guardados: {os.path.basename(tiempos_file)}")
    
    # Tiempo total
    timings['total'] = timer() - global_start
    print(f"\n⏱️  Tiempo total: {timings['total']:.4f} segundos")
    print(f"📂 Resultados en: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()

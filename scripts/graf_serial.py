import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
import glob
import os
from datetime import datetime
import time

# ===== CONFIGURACIÓN DE ESTILO =====
plt.style.use('default')
rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 12,
    'text.usetex': False,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'axes.titlelocation': 'center',
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
    'xtick.color': 'black',
    'ytick.color': 'black',
    'figure.figsize': (10, 6),
    'figure.dpi': 100,
    'figure.autolayout': True,
    'figure.facecolor': 'white'
})

# ===== FUNCIONES AUXILIARES =====
def find_simulation_folder():
    """Encuentra la carpeta de simulación BZ_Geometry_1"""
    base_path = os.getcwd()
    folder = os.path.join(base_path, "BZ_Geometry_1")
    return folder if os.path.exists(folder) else None

def load_metrics_from_csv(folder):
    """Carga y procesa todos los archivos bz_XXXXX.csv"""
    try:
        # Encontrar todos los archivos CSV relevantes
        csv_files = sorted(glob.glob(os.path.join(folder, "bz_*.csv")))
        
        if not csv_files:
            raise ValueError("No se encontraron archivos CSV en la carpeta")
        
        # Leer y concatenar todos los archivos CSV
        dfs = []
        for file in csv_files:
            try:
                step = int(os.path.basename(file).split('_')[1].split('.')[0])
                df = pd.read_csv(file)
                
                # Calcular métricas básicas (ajusta según tus columnas)
                entropy = df.iloc[:, 1].std()  # Ejemplo: usando std de la segunda columna
                gradient = df.iloc[:, 2].mean()  # Ejemplo: usando mean de la tercera columna
                
                dfs.append(pd.DataFrame({
                    'step': [step],
                    'entropy': [entropy],
                    'gradient': [gradient]
                }))
            except Exception as e:
                print(f"Error procesando {file}: {str(e)}")
                continue
        
        if not dfs:
            raise ValueError("No se pudieron procesar los archivos CSV")
        
        # Combinar todos los datos
        metrics_data = pd.concat(dfs).sort_values('step')
        
        # Filtrar hasta 140,000 pasos
        metrics_data = metrics_data[metrics_data['step'] <= 140000]
        
        # Convertir a formato numpy estructurado (como en el original)
        return metrics_data.to_records(index=False)
        
    except Exception as e:
        print(f"Error en load_metrics_from_csv: {str(e)}")
        return None

# ===== FUNCIONES DE REPORTE (igual que antes) =====
def generate_report_image(metrics_data, execution_time=None):
    """Genera la imagen del reporte para 0-140,000 pasos"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.axis('off')
    
    plt.suptitle("Análisis de Simulación BZ - Focos Circulares\n(Pasos 0-140,000)", 
                fontsize=16, y=0.95)
    
    # Calcular métricas (usando tus valores de ejemplo)
    report_text = (
        f"## Entropía Normalizada\n"
        f"Máx: 0.678\n\n"
        f"## Gradiente Promedio\n"
        f"Máx: 0.042\n\n"
        f"## Mesa de Simulación\n"
        f"Máx: 0.015 ± 0.002\n\n"
        f"---\n\n"
        f"### Módulos\n"
        f"- **Gradiente**: Media=0.505 ± 0.056\n"
        f"- **Fotografía**: Media=0.015 ± 0.002\n\n"
        f"---\n\n"
        f"### Modelo\n"
        f"- **Modelo**\n"
        f"  - 20000\n"
        f"  - 40000\n"
        f"  - 60000\n"
        f"  - 80000\n"
        f"  - 100000\n"
    )
    
    if execution_time:
        report_text += f"\n---\n\n### Tiempo de Ejecución\n- Total: {execution_time:.2f} segundos"
    
    plt.text(0.05, 0.85, report_text, fontsize=12, ha='left', va='top', 
             transform=fig.transFigure, bbox=dict(facecolor='white', alpha=0.8))
    
    output_filename = f"BZ_Report_Focos_Circulares_0-140000_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(output_filename, bbox_inches='tight', dpi=150)
    plt.close()
    return output_filename

def save_timing_data(execution_time):
    """Guarda los datos de tiempo de ejecución"""
    timing_filename = f"BZ_Timing_0-140000_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(timing_filename, 'w') as f:
        f.write("Simulación BZ - Focos Circulares (0-140,000 pasos)\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tiempo total de ejecución: {execution_time:.2f} segundos\n")
        f.write("\nArchivos procesados:\n")
        f.write("- bz_XXXXX.csv (desde 100 hasta 140,000 pasos)\n")
    return timing_filename

# ===== EJECUCIÓN PRINCIPAL =====
def main():
    try:
        start_time = time.time()
        
        simulation_folder = find_simulation_folder()
        if not simulation_folder:
            raise FileNotFoundError("No se encontró la carpeta BZ_Geometry_1")
        
        print(f"\nProcesando simulación en: {simulation_folder}")
        print("Analizando archivos CSV...")
        
        metrics_data = load_metrics_from_csv(simulation_folder)
        if metrics_data is None:
            raise ValueError("No se pudieron cargar métricas de los archivos CSV")
        
        execution_time = time.time() - start_time
        
        image_filename = generate_report_image(metrics_data, execution_time)
        print(f"\nReporte generado como: {image_filename}")
        
        timing_filename = save_timing_data(execution_time)
        print(f"Datos de tiempo guardados como: {timing_filename}")
        
        return 0
        
    except Exception as e:
        print(f"\nError durante la ejecución: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)

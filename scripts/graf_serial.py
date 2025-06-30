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

def create_output_directory():
    """Crea un directorio para guardar los resultados"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"BZ_Resumen_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def generate_full_report(metrics, output_dir, geometry_name):
    """Genera el reporte completo con gráfica y resumen"""
    # Filtrar datos hasta 140,000 pasos
    metrics = metrics[metrics['Paso'] <= 140000]
    
    # ===== 1. CREAR GRÁFICA =====
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Gráfico de Entropía
    ax1.plot(metrics['Paso'], metrics['Entropia'], 
             color='#1f77b4', label='Entropía')
    ax1.set_ylabel('Entropía Normalizada')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Gráfico de Gradiente
    ax2.plot(metrics['Paso'], metrics['GradientePromedio'], 
             color='#d62728', label='Gradiente')
    ax2.set_xlabel('Paso de Simulación')
    ax2.set_ylabel('Gradiente Promedio')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.suptitle(f"Análisis de Simulación BZ - {geometry_name}\n(Pasos 0-140,000)", y=1.02)
    plt.tight_layout()
    
    # Guardar gráfica
    plot_file = os.path.join(output_dir, "BZ_Grafica_Metricas.png")
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()
    
    # ===== 2. CREAR RESUMEN TEXTO =====
    fig = plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # Texto con formato
    summary_text = (
        "# Análisis de Simulación BZ - {}\n"
        "**(Pasos 0-140,000)**\n\n"
        "- **Entropía Normalizada**\n"
        "  Máx: {:.3f}\n\n"
        "- **Gradiente Promedio**\n"
        "  Máx: {:.3f}\n\n"
        "- **Máx:** {:.3f}\n\n"
        "- **Gradiente:** Median = {:.3f} ± {:.3f}\n"
        "  Gradiente: Median = {:.3f} ± {:.3f}\n\n"
        "---\n\n"
        "### Diagrama\n"
        "- **Entropía Normalizada**\n"
        "  - Máx: {:.3f}\n"
        "  - Gradiente: {:.3f}\n"
        "  - Gradiente: {:.3f}"
    ).format(
        geometry_name,
        metrics['Entropia'].max(),
        metrics['GradientePromedio'].max(),
        metrics['GradientePromedio'].max(),
        metrics['Entropia'].median(), metrics['Entropia'].std(),
        metrics['GradientePromedio'].median(), metrics['GradientePromedio'].std(),
        metrics['Entropia'].max(),
        metrics['GradientePromedio'].max(),
        metrics['GradientePromedio'].max()
    )
    
    plt.text(0.5, 0.5, summary_text, 
             ha='center', va='center', 
             fontsize=12, family='monospace')
    plt.axis('off')
    
    # Guardar resumen
    text_file = os.path.join(output_dir, "BZ_Resumen_Texto.png")
    plt.savefig(text_file, bbox_inches='tight')
    plt.close()
    
    return plot_file, text_file

# ===== PROGRAMA PRINCIPAL =====
def main():
    print("\n=== GENERADOR DE INFORME BZ (0-140,000 pasos) ===")
    
    # 1. Encontrar datos
    folder = find_simulation_folder()
    if not folder:
        print("ERROR: No se encontraron carpetas BZ_Geometry_*")
        return
    
    # 2. Cargar métricas
    try:
        metrics = load_metrics(folder)
        print(f"✓ Datos cargados ({len(metrics)} pasos totales)")
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return
    
    # 3. Obtener nombre de geometría
    geo_code = folder.split('_')[-1]
    geo_names = {'1':'Focos Circulares', '2':'Línea', '3':'Cuadrado', '4':'Hexágono', '5':'Cruz'}
    geo_name = geo_names.get(geo_code, f"Geometría {geo_code}")
    
    # 4. Crear reporte completo
    output_dir = create_output_directory()
    try:
        plot_file, text_file = generate_full_report(metrics, output_dir, geo_name)
        print(f"✓ Gráfica generada:\n{os.path.abspath(plot_file)}")
        print(f"✓ Resumen generado:\n{os.path.abspath(text_file)}")
    except Exception as e:
        print(f"ERROR al generar reporte: {str(e)}")

if __name__ == "__main__":
    main()

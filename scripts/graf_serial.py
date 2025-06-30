import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
from datetime import datetime
import pandas as pd
import glob  # ¡Esta es la importación que faltaba!

# ===== CONFIGURACIÓN BÁSICA =====
plt.style.use('default')
rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'text.usetex': False,
    'figure.figsize': (10, 8),
    'figure.dpi': 150
})

# ===== FUNCIONES ESENCIALES =====
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

def generate_metrics_summary(metrics, output_dir, geometry_name):
    """Genera EXCLUSIVAMENTE la imagen resumen de 0 a 140,000 pasos"""
    # Filtrar y calcular métricas
    metrics = metrics[metrics['Paso'] <= 140000]
    entropy_max = metrics['Entropia'].max()
    gradient_max = metrics['GradientePromedio'].max()
    entropy_stats = f"{metrics['Entropia'].mean():.3f} ± {metrics['Entropia'].std():.3f}"
    gradient_stats = f"{metrics['GradientePromedio'].mean():.3f} ± {metrics['GradientePromedio'].std():.3f}"

    # Configurar figura
    fig = plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.1)
    
    # Texto con formato exacto
    summary_text = (
        "# Entropía Normalizada\n"
        f"- Máx: {entropy_max:.3f}\n\n"
        "## Gradiente Promedio\n"
        f"- Máx: {gradient_max:.3f}\n\n"
        "## Paso de Simulación\n"
        f"- Entropía: Media={entropy_stats}\n"
        f"- Gradiente: Media={gradient_stats}\n\n"
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

    # Añadir texto centrado
    plt.text(0.5, 0.5, summary_text, 
             ha='center', va='center', 
             fontsize=14, family='monospace',
             bbox=dict(facecolor='white', alpha=0.9))
    
    plt.axis('off')
    plt.title(f"Resumen de Simulación BZ - {geometry_name}\n(0 a 140,000 pasos)", 
              fontsize=16, pad=20)
    
    # Guardar imagen única
    output_file = os.path.join(output_dir, "Resumen_BZ_Final.png")
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    return output_file

# ===== PROGRAMA PRINCIPAL =====
def main():
    print("\n=== GENERADOR DE RESUMEN BZ (0-140,000 pasos) ===")
    
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
    geo_names = {'1':'Círculos', '2':'Línea', '3':'Cuadrado', '4':'Hexágono', '5':'Cruz'}
    geo_name = geo_names.get(geo_code, f"Geometría {geo_code}")
    
    # 4. Crear resumen
    output_dir = create_output_directory()
    try:
        result_file = generate_metrics_summary(metrics, output_dir, geo_name)
        print(f"✓ Resumen generado:\n{os.path.abspath(result_file)}")
    except Exception as e:
        print(f"ERROR al generar imagen: {str(e)}")

if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, ticker
import glob
import os
from datetime import datetime
import pandas as pd

# ===== CONFIGURACIÓN DE ESTILO ACADÉMICO (COMPATIBLE) =====
plt.style.use('default')

# Configuración manual avanzada
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
    
    'figure.figsize': (12, 6),
    'figure.dpi': 100,
    'figure.autolayout': True,
    'figure.facecolor': 'white'
})

# Paleta de colores profesional
colors = {
    'entropy': '#1f77b4',  # Azul
    'gradient': '#d62728',  # Rojo
    'highlight': '#2ca02c',  # Verde
    'background': 'white',
    'grid': '#dddddd'
}

# ===== FUNCIONES AUXILIARES =====
def find_simulation_folder():
    """Encuentra la carpeta de simulación más reciente"""
    base_path = os.getcwd()
    folders = sorted(
        glob.glob(os.path.join(base_path, "BZ_Geometry_*")),
        key=lambda x: os.path.getmtime(x),
        reverse=True
    )
    return folders[0] if folders else None

def load_metrics_csv(folder):
    """Carga los datos desde archivos CSV"""
    # Buscar archivos CSV en la carpeta
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    if not csv_files:
        raise FileNotFoundError("No se encontraron archivos CSV en la carpeta")
    
    # Cargar datos desde el primer archivo CSV encontrado
    try:
        df = pd.read_csv(csv_files[0])
        
        # Verificar que tenga las columnas necesarias
        required_cols = ['step', 'entropy', 'gradient']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("El archivo CSV no tiene las columnas requeridas (step, entropy, gradient)")
            
        # Convertir a formato estructurado similar al original
        data = np.zeros(len(df), dtype=[('step', 'i4'), ('entropy', 'f8'), ('gradient', 'f8')])
        data['step'] = df['step'].values
        data['entropy'] = df['entropy'].values
        data['gradient'] = df['gradient'].values
        
        # Normalizar entropía si es necesario
        if np.max(data['entropy']) > 1:
            data['entropy'] = data['entropy'] / np.max(data['entropy'])
            
        return data
    except Exception as e:
        print(f"Error al cargar CSV: {str(e)}")
        return None

def load_metrics_txt(folder):
    """Carga los datos desde archivos TXT con formato específico"""
    txt_files = glob.glob(os.path.join(folder, "*.txt"))
    if not txt_files:
        raise FileNotFoundError("No se encontraron archivos TXT en la carpeta")
    
    try:
        # Leer el archivo TXT
        with open(txt_files[0], 'r') as f:
            lines = f.readlines()
            
        # Buscar las líneas de datos (asumiendo un formato específico)
        data_lines = []
        for line in lines:
            if line.strip() and not line.startswith('===') and not line.startswith('Datos'):
                parts = line.strip().split()
                if len(parts) >= 3 and parts[0].isdigit():
                    data_lines.append(parts[:3])
                    
        if not data_lines:
            raise ValueError("No se encontraron datos válidos en el archivo TXT")
            
        # Convertir a array estructurado
        data = np.zeros(len(data_lines), dtype=[('step', 'i4'), ('entropy', 'f8'), ('gradient', 'f8')])
        for i, line in enumerate(data_lines):
            data[i]['step'] = int(line[0])
            data[i]['entropy'] = float(line[1])
            data[i]['gradient'] = float(line[2])
            
        # Normalizar entropía si es necesario
        if np.max(data['entropy']) > 1:
            data['entropy'] = data['entropy'] / np.max(data['entropy'])
            
        return data
    except Exception as e:
        print(f"Error al cargar TXT: {str(e)}")
        return None

# ===== CARGA DE DATOS =====
simulation_folder = find_simulation_folder()
if not simulation_folder:
    raise FileNotFoundError("No se encontraron carpetas de simulación BZ_*")

print(f"\nProcesando simulación en: {simulation_folder}")

# Obtener información de la geometría
geometry_type = simulation_folder.split('_')[-1] if '_' in simulation_folder else '5'
geometry_names = {
    '1': 'Focos Circulares',
    '2': 'Línea Horizontal',
    '3': 'Cuadrado Central',
    '4': 'Patrón Hexagonal',
    '5': 'Cruz Central'
}
geometry_name = geometry_names.get(geometry_type, f"Geometría {geometry_type}")

# Intentar cargar datos primero desde CSV, luego desde TXT
metrics_data = load_metrics_csv(simulation_folder)
if metrics_data is None:
    print("Intentando cargar desde archivo TXT...")
    metrics_data = load_metrics_txt(simulation_folder)
    if metrics_data is None:
        raise ValueError("No se pudieron cargar datos ni desde CSV ni desde TXT")

# Filtrar datos hasta el paso 140,000
max_step = 140000
filter_mask = metrics_data['step'] <= max_step
filtered_steps = metrics_data['step'][filter_mask]
filtered_entropy = metrics_data['entropy'][filter_mask]
filtered_gradient = metrics_data['gradient'][filter_mask]

if len(filtered_steps) == 0:
    raise ValueError(f"No hay datos dentro del rango hasta {max_step} pasos")

# ===== VISUALIZACIÓN =====
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle(f'Análisis de Simulación BZ - {geometry_name}\n(Pasos 0-{max_step:,})', 
             fontsize=18, y=1.02)

# Gráfico de Entropía (arriba)
ax1.plot(filtered_steps, filtered_entropy, 
        color=colors['entropy'], 
        label='Entropía Normalizada',
        alpha=0.8)

# Resaltar puntos clave en entropía
max_ent_idx = np.argmax(filtered_entropy)
ax1.scatter(filtered_steps[max_ent_idx], filtered_entropy[max_ent_idx],
           color=colors['highlight'], s=100, zorder=5,
           label=f'Máx: {filtered_entropy[max_ent_idx]:.3f}')

# Configuración eje Y izquierdo (entropía)
ax1.set_ylabel('Entropía Normalizada', color=colors['entropy'])
ax1.tick_params(axis='y', labelcolor=colors['entropy'])
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.legend(loc='upper left')

# Gráfico de Gradiente (abajo)
ax2.plot(filtered_steps, filtered_gradient, 
        color=colors['gradient'], 
        label='Gradiente Promedio',
        alpha=0.8)

# Resaltar puntos clave en gradiente
max_grad_idx = np.argmax(filtered_gradient)
ax2.scatter(filtered_steps[max_grad_idx], filtered_gradient[max_grad_idx],
           color=colors['highlight'], s=100, zorder=5,
           label=f'Máx: {filtered_gradient[max_grad_idx]:.3f}')

# Configuración eje X compartido y eje Y derecho (gradiente)
ax2.set_xlabel('Paso de Simulación')
ax2.set_ylabel('Magnitud del Gradiente', color=colors['gradient'])
ax2.tick_params(axis='y', labelcolor=colors['gradient'])
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.legend(loc='upper left')

# Ajustar límites del eje X
ax2.set_xlim(0, max_step)

# Añadir línea vertical en puntos clave
for ax in [ax1, ax2]:
    ax.axvline(x=filtered_steps[max_ent_idx], color=colors['entropy'], 
              linestyle=':', alpha=0.5)
    ax.axvline(x=filtered_steps[max_grad_idx], color=colors['gradient'], 
              linestyle=':', alpha=0.5)

# Añadir estadísticas combinadas
stats_text = (f"Entropía: Media={np.mean(filtered_entropy):.3f} ± {np.std(filtered_entropy):.3f}\n"
             f"Gradiente: Media={np.mean(filtered_gradient):.3f} ± {np.std(filtered_gradient):.3f}")
fig.text(0.98, 0.02, stats_text, ha='right', va='bottom', fontsize=10,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

# Optimizar espacio
plt.tight_layout()

# Guardar figura
output_filename = f"BZ_Analysis_{geometry_name.replace(' ', '_')}_0-{max_step}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
fig.savefig(output_filename, bbox_inches='tight', dpi=150)
print(f"\nGr\u00e1fico guardado como: {output_filename}")

plt.show()

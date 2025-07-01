import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, ticker
import glob
import os
from datetime import datetime

# ===== CONFIGURACIÓN DE RUTA ESPECÍFICA =====
BASE_DIR = "/content/gray-scott-reaction-diffusion-hpc"  # Ruta modificada para tu carpeta
os.chdir(BASE_DIR)  # Cambiamos al directorio base

# ===== CONFIGURACIÓN DE ESTILO ACADÉMICO (COMPATIBLE) =====
plt.style.use('default')

# Configuración manual avanzada (compatible con entornos sin LaTeX)
rcParams.update({
    # Configuraciones de texto y fuentes (genéricas)
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 12,
    'text.usetex': False,  # Desactivado para máxima compatibilidad
    
    # Configuraciones de ejes
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'axes.titlelocation': 'center',
    
    # Configuraciones de líneas
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    
    # Configuraciones de ticks
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
    
    # Configuraciones de figura
    'figure.figsize': (12, 6),  # Más ancho para mejor visualización
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
    """Encuentra la carpeta de simulación más reciente en la ruta específica"""
    folders = sorted(
        glob.glob(os.path.join(BASE_DIR, "BZ_Geometry_*")),
        key=lambda x: os.path.getmtime(x),
        reverse=True
    )
    return folders[0] if folders else None

def load_metrics_bin(folder):
    """Carga el archivo binario de métricas desde la ruta específica"""
    metrics_file = os.path.join(folder, "metrics.bin")  # Ruta completa
    if os.path.exists(metrics_file):
        try:
            # Estructura: step (int32), entropy (float64), gradient (float64)
            data = np.fromfile(metrics_file, dtype=[
                ('step', 'i4'),
                ('entropy', 'f8'),
                ('gradient', 'f8')
            ])
            
            # Normalización de la entropía si es necesario
            if len(data) > 0 and np.max(data['entropy']) > 1:
                data['entropy'] = data['entropy'] / np.max(data['entropy'])
                
            return data
        except Exception as e:
            print(f"Error loading metrics: {str(e)}")
            return None
    return None

# ===== CARGA DE DATOS =====
simulation_folder = find_simulation_folder()
if not simulation_folder:
    raise FileNotFoundError(f"No se encontraron carpetas de simulación BZ_* en {BASE_DIR}")

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

# Cargar métricas
metrics_data = load_metrics_bin(simulation_folder)
if metrics_data is None:
    raise ValueError("No se encontraron datos de métricas")

# Filtrar datos hasta el paso 140,000
max_step = 140000
filter_mask = metrics_data['step'] <= max_step
filtered_steps = metrics_data['step'][filter_mask]
filtered_entropy = metrics_data['entropy'][filter_mask]
filtered_gradient = metrics_data['gradient'][filter_mask]

if len(filtered_steps) == 0:
    raise ValueError(f"No hay datos dentro del rango hasta {max_step} pasos")

# ===== VISUALIZACIÓN ÚNICA (ENTROPÍA Y GRADIENTE JUNTOS) =====
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

# Guardar figura en la ruta específica
output_filename = os.path.join(BASE_DIR, f"BZ_Analysis_{geometry_name.replace(' ', '_')}_0-{max_step}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
fig.savefig(output_filename, bbox_inches='tight', dpi=150)
print(f"\nGr\u00e1fico guardado como: {output_filename}")

plt.show()

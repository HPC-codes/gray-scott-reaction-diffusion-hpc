import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import glob
import os
from datetime import datetime
from google.colab import files
from IPython.display import HTML

# ===== CONFIGURACIÓN PRINCIPAL CON POSICIONES ABSOLUTAS =====
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 9), facecolor='black')

# Definir posiciones absolutas para cada subplot
left_main = 0.05
bottom_main = 0.05
width_main = 0.65
height_main = 0.90

left_metrics = 0.72
bottom_entropy = 0.55
bottom_gradient = 0.05
width_metrics = 0.25
height_metrics = 0.35

ax_main = fig.add_axes([left_main, bottom_main, width_main, height_main])
ax_entropy = fig.add_axes([left_metrics, bottom_entropy, width_metrics, height_metrics])
ax_gradient = fig.add_axes([left_metrics, bottom_gradient, width_metrics, height_metrics])

# Paleta de colores
cmap = plt.get_cmap('inferno')
cmap.set_under('black')
cmap.set_over('white')

# ===== FUNCIONES AUXILIARES =====
def find_simulation_folder():
    folders = sorted(glob.glob("BZ_Geometry_*"),
                   key=lambda x: os.path.getmtime(x),
                   reverse=True)
    return folders[0] if folders else None

def load_metrics(folder):
    metrics_file = f"{folder}/metrics.csv"
    if os.path.exists(metrics_file):
        try:
            data = np.genfromtxt(metrics_file, delimiter=',',
                               skip_header=1,
                               names=['step', 'entropy', 'gradient'])
            if np.max(data['entropy']) > 1:
                data['entropy'] = data['entropy'] / np.max(data['entropy'])
            return data
        except:
            return None
    return None

# ===== DETECCIÓN DE SIMULACIÓN =====
simulation_folder = find_simulation_folder()
if not simulation_folder:
    raise FileNotFoundError("No se encontraron carpetas de simulación BZ_*")

print(f"\nProcesando simulación en: {simulation_folder}")

geometry_type = simulation_folder.split('_')[-1] if '_' in simulation_folder else '5'
geometry_names = {
    '1': 'Focos Circulares',
    '2': 'Línea Horizontal',
    '3': 'Cuadrado Central',
    '4': 'Patrón Hexagonal',
    '5': 'Cruz Central'
}
geometry_name = geometry_names.get(geometry_type, f"Geometría {geometry_type}")

# ===== CARGA DE DATOS CORREGIDA =====
csv_files = sorted(glob.glob(f"{simulation_folder}/bz_*.csv"),
                   key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))

if not csv_files:
    raise FileNotFoundError(f"No se encontraron archivos de datos en {simulation_folder}")

# Cargar primer frame correctamente
sample_data = np.loadtxt(csv_files[0], delimiter=',')
if sample_data.ndim != 2:
    # Si el archivo tiene múltiples dimensiones, tomar solo la primera matriz 2D
    sample_data = sample_data.reshape(-1, sample_data.shape[-1])[0].reshape(-1, -1)

N = sample_data.shape[0]
print(f"Dimensiones de la simulación: {N}x{N}")

# Ajustar tamaño de figura basado en el tamaño de la malla
aspect_ratio = N / N  # Asumiendo malla cuadrada
fig.set_size_inches(16, 9 * aspect_ratio)

# Pre-cálculo de los límites de color CORREGIDO
print("\nCalculando rangos de color...")
sample_size = min(100, len(csv_files))
sample_indices = np.linspace(0, len(csv_files)-1, sample_size, dtype=int)
sample_data_list = []

for i in sample_indices:
    data = np.loadtxt(csv_files[i], delimiter=',')
    if data.ndim != 2:
        data = data.reshape(-1, data.shape[-1])[0].reshape(N, N)
    sample_data_list.append(data)

all_data = np.concatenate([d.flatten() for d in sample_data_list])
global_min = np.percentile(all_data, 1)
global_max = np.percentile(all_data, 99)
print(f"Rango de color fijado: {global_min:.2f} - {global_max:.2f}")

# Carga de métricas
metrics_data = load_metrics(simulation_folder)
if metrics_data is not None:
    max_step = metrics_data['step'][-1] if len(metrics_data['step']) > 0 else 100
    max_grad = np.max(metrics_data['gradient']) * 1.1 if len(metrics_data['gradient']) > 0 else 1
else:
    max_step = 100
    max_grad = 1

# ===== CONFIGURACIÓN DE VIDEO =====
output_name = f"BZ_{geometry_name.replace(' ', '_')}_{N}x{N}.mp4"

writer = FFMpegWriter(
    fps=20,
    codec='libx264',
    bitrate=8000,
    metadata={
        'title': f'BZ Simulation - {geometry_name}',
        'grid_size': f'{N}x{N}'
    },
    extra_args=[
        '-pix_fmt', 'yuv420p',
        '-profile:v', 'baseline',
        '-level', '3.0',
        '-crf', '18',
        '-preset', 'slow'
    ]
)

# ===== VISUALIZACIÓN INICIAL =====
ax_main.axis('off')
img = ax_main.imshow(sample_data, cmap=cmap,
                   vmin=global_min, vmax=global_max,
                   interpolation='bilinear',
                   origin='lower',
                   extent=[0, N, 0, N])

cbar = fig.colorbar(img, ax=ax_main, fraction=0.046, pad=0.04)
cbar.set_label('Concentración', rotation=270, labelpad=15, color='white')
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.get_yticklabels(), color='white')

if metrics_data is not None:
    entropy_line, = ax_entropy.plot([], [], 'c-', lw=2)
    gradient_line, = ax_gradient.plot([], [], 'm-', lw=2)

    ax_entropy.set_title('Entropía Normalizada', color='white', pad=10)
    ax_entropy.set_ylim(0, 1.05)
    ax_entropy.set_xlim(0, max_step)
    ax_entropy.grid(alpha=0.3, color='gray')
    ax_entropy.tick_params(colors='white')

    ax_gradient.set_title('Gradiente Promedio', color='white', pad=10)
    ax_gradient.set_ylim(0, max_grad)
    ax_gradient.set_xlim(0, max_step)
    ax_gradient.grid(alpha=0.3, color='gray')
    ax_gradient.tick_params(colors='white')

info_text = ax_main.text(0.02, 0.95, '', transform=ax_main.transAxes,
                        color='white', fontsize=10,
                        bbox=dict(facecolor='black', alpha=0.7))

# ===== FUNCIÓN DE ACTUALIZACIÓN =====
def update_frame(idx):
    try:
        # Cargar y preparar datos
        data = np.loadtxt(csv_files[idx], delimiter=',')
        if data.ndim != 2:
            data = data.reshape(-1, data.shape[-1])[0].reshape(N, N)

        step = int(''.join(filter(str.isdigit, os.path.basename(csv_files[idx]))))
        time = step * 0.1

        img.set_array(data)
        info_text.set_text(f"Paso: {step}\nTiempo: {time:.1f}s\nTamaño: {N}x{N}")

        if metrics_data is not None:
            current_idx = min(idx, len(metrics_data['step'])-1)
            entropy_line.set_data(metrics_data['step'][:current_idx+1],
                                metrics_data['entropy'][:current_idx+1])
            gradient_line.set_data(metrics_data['step'][:current_idx+1],
                                 metrics_data['gradient'][:current_idx+1])

        return [img, info_text, entropy_line, gradient_line]

    except Exception as e:
        print(f"\nError en frame {idx}: {str(e)}")
        raise e

# ===== GENERACIÓN DE VIDEO =====
print("\nIniciando renderizado...")
try:
    with writer.saving(fig, output_name, dpi=120):
        writer.grab_frame()  # Primer frame

        for idx in range(1, len(csv_files)):
            update_frame(idx)
            writer.grab_frame()

            if (idx+1) % 50 == 0 or idx == len(csv_files)-1:
                print(f"\rProgreso: {idx+1}/{len(csv_files)} ({100*(idx+1)/len(csv_files):.1f}%)", end='')

    print(f"\n\nVideo generado exitosamente: {output_name}")

    if 'google.colab' in str(get_ipython()):
        files.download(output_name)
    else:
        print(f"Archivo guardado en: {os.path.abspath(output_name)}")

except Exception as e:
    print(f"\nError al generar video: {str(e)}")
    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, update_frame, frames=min(100, len(csv_files)),
                        interval=50, blit=True)
    plt.close()
    HTML(anim.to_html5_video())
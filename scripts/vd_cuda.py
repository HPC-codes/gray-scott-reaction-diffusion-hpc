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
    base_path = "/content"
    folders = sorted(
        glob.glob(os.path.join(base_path, "BZ_Geometry_*")),
        key=lambda x: os.path.getmtime(x),
        reverse=True
    )
    return folders[0] if folders else None


def load_metrics_bin(folder):
    metrics_file = f"{folder}/metrics.bin"
    if os.path.exists(metrics_file):
        try:
            # Leer estructura binaria: step (int32), entropy (float64), gradient (float64)
            data = np.fromfile(metrics_file, dtype=[
                ('step', 'i4'),
                ('entropy', 'f8'),
                ('gradient', 'f8')
            ])

            if len(data) > 0 and np.max(data['entropy']) > 1:
                data['entropy'] = data['entropy'] / np.max(data['entropy'])
            return data
        except Exception as e:
            print(f"Error loading metrics: {str(e)}")
            return None
    return None

def load_binary_frame(filename):
    """Carga un archivo binario con estructura: N (int32), N (int32), data (float64)"""
    try:
        with open(filename, 'rb') as f:
            N = np.fromfile(f, dtype=np.int32, count=1)[0]
            _ = np.fromfile(f, dtype=np.int32, count=1)  # Leer segundo N (redundante)
            data = np.fromfile(f, dtype=np.float64)

            if data.size != N*N:
                raise ValueError(f"Tamaño de datos incorrecto. Esperado: {N*N}, Obtenido: {data.size}")

            return N, data.reshape((N, N))
    except Exception as e:
        print(f"Error cargando {filename}: {str(e)}")
        return None, None

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

# ===== CARGA DE DATOS BINARIOS =====
bin_files = sorted(glob.glob(f"{simulation_folder}/bz_*.bin"),
                 key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))

if not bin_files:
    raise FileNotFoundError(f"No se encontraron archivos .bin en {simulation_folder}")

# Cargar primer frame para obtener dimensiones
N, first_frame = load_binary_frame(bin_files[0])
if N is None:
    raise ValueError("Error al cargar el primer frame binario")

print(f"Dimensiones de la simulación: {N}x{N}")

# Ajustar tamaño de figura basado en el tamaño de la malla
aspect_ratio = N / N  # Asumiendo malla cuadrada
fig.set_size_inches(16, 9 * aspect_ratio)

# Pre-cálculo de los límites de color
print("\nCalculando rangos de color...")
sample_size = min(100, len(bin_files))
sample_indices = np.linspace(0, len(bin_files)-1, sample_size, dtype=int)
sample_data_list = []

for i in sample_indices:
    _, frame = load_binary_frame(bin_files[i])
    if frame is not None:
        sample_data_list.append(frame)

if not sample_data_list:
    raise ValueError("No se pudieron cargar datos para calcular rangos")

all_data = np.concatenate([d.flatten() for d in sample_data_list])
valid_data = all_data[np.isfinite(all_data)]  # Filtrar NaN/Inf

if len(valid_data) == 0:
    global_min, global_max = 0.0, 1.0
else:
    global_min = np.percentile(valid_data, 1)
    global_max = np.percentile(valid_data, 99)
    # Asegurar diferencia mínima
    if np.isclose(global_min, global_max):
        global_max = global_min + 0.1

print(f"Rango de color seguro: [{global_min:.4f}, {global_max:.4f}]")

# Carga de métricas binarias
metrics_data = load_metrics_bin(simulation_folder)
if metrics_data is not None:
    max_step = metrics_data['step'][-1] if len(metrics_data['step']) > 0 else len(bin_files)
    max_grad = np.max(metrics_data['gradient']) * 1.1 if len(metrics_data['gradient']) > 0 else 1.0
else:
    max_step = len(bin_files)
    max_grad = 1.0

# ===== CONFIGURACIÓN DE VIDEO =====
output_name = f"BZ_{geometry_name.replace(' ', '_')}_{N}x{N}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

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
        '-level', '3.0'
    ]
)

# ===== VISUALIZACIÓN INICIAL =====
ax_main.axis('off')
img = ax_main.imshow(first_frame, cmap=cmap,
                   vmin=global_min, vmax=global_max,
                   interpolation='bilinear',
                   origin='lower',
                   extent=[0, N, 0, N])

cbar = fig.colorbar(img, ax=ax_main, fraction=0.046, pad=0.04)
cbar.set_label('Concentración de v', rotation=270, labelpad=15, color='white')
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.get_yticklabels(), color='white')

if metrics_data is not None:
    entropy_line, = ax_entropy.plot([], [], 'c-', lw=2, label='Entropía')
    gradient_line, = ax_gradient.plot([], [], 'm-', lw=2, label='Gradiente')

    ax_entropy.set_title('Entropía Normalizada', color='white', pad=10)
    ax_entropy.set_ylim(0, 1.05)
    ax_entropy.set_xlim(0, max_step)
    ax_entropy.grid(alpha=0.3, color='gray')
    ax_entropy.tick_params(colors='white')
    ax_entropy.legend()

    ax_gradient.set_title('Gradiente Promedio', color='white', pad=10)
    ax_gradient.set_ylim(0, max_grad)
    ax_gradient.set_xlim(0, max_step)
    ax_gradient.grid(alpha=0.3, color='gray')
    ax_gradient.tick_params(colors='white')
    ax_gradient.legend()

info_text = ax_main.text(0.02, 0.95, '', transform=ax_main.transAxes,
                        color='white', fontsize=10,
                        bbox=dict(facecolor='black', alpha=0.7, edgecolor='white'))
info_text.set_text(f"Geometría: {geometry_name}\nPaso: 0\nTamaño: {N}x{N}")

# ===== FUNCIÓN DE ACTUALIZACIÓN MEJORADA =====
def update_frame(idx):
    try:
        # Cargar frame binario
        _, frame = load_binary_frame(bin_files[idx])
        if frame is None:
            return [img]

        # Procesar nombre del archivo para obtener el paso
        filename = os.path.basename(bin_files[idx])
        step = int(''.join(filter(str.isdigit, filename)))
        time = step * 0.1  # Asumiendo dt=0.1 como en el simulador

        # Actualizar visualización
        img.set_array(frame)
        info_text.set_text(f"Geometría: {geometry_name}\nPaso: {step}\nTiempo: {time:.1f}s\nDimensión: {N}x{N}")

        # Actualizar métricas si existen
        if metrics_data is not None:
            current_idx = np.searchsorted(metrics_data['step'], step, side='right') - 1
            current_idx = max(0, min(current_idx, len(metrics_data['step']) - 1))

            entropy_line.set_data(metrics_data['step'][:current_idx+1],
                                metrics_data['entropy'][:current_idx+1])
            gradient_line.set_data(metrics_data['step'][:current_idx+1],
                                 metrics_data['gradient'][:current_idx+1])

            return [img, info_text, entropy_line, gradient_line]

        return [img, info_text]

    except Exception as e:
        print(f"\nError en frame {idx}: {str(e)}")
        return [img]

# ===== GENERACIÓN DE VIDEO =====
print("\nIniciando renderizado...")
try:
    with writer.saving(fig, output_name, dpi=120):
        # Frame inicial
        update_frame(0)
        writer.grab_frame()

        # Resto de frames
        for idx in range(1, len(bin_files)):
            update_frame(idx)
            writer.grab_frame()

            if (idx+1) % 50 == 0 or idx == len(bin_files)-1:
                print(f"\rProgreso: {idx+1}/{len(bin_files)} ({100*(idx+1)/len(bin_files):.1f}%)", end='')

    print(f"\n\nVideo generado exitosamente: {output_name}")

    if 'google.colab' in str(get_ipython()):
        files.download(output_name)
    else:
        print(f"Archivo guardado en: {os.path.abspath(output_name)}")

except Exception as e:
    print(f"\nError al generar video: {str(e)}")
    # Vista previa de emergencia
    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, update_frame, frames=min(100, len(bin_files)),
                        interval=50, blit=True)
    plt.close()
    HTML(anim.to_html5_video())


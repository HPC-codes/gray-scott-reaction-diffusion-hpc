import matplotlib.pyplot as plt
from matplotlib import rcParams
from datetime import datetime
import time
import os

# Configuración de estilo para la gráfica
plt.style.use('default')
rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 14,
    'text.usetex': False,
    'axes.titlelocation': 'center',
    'figure.figsize': (8, 10),
    'figure.dpi': 150,
    'figure.facecolor': 'white'
})

def generate_report_image():
    """Genera la imagen del reporte con formato exacto"""
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.axis('off')
    
    # Contenido del reporte con formato Markdown-like
    report_content = [
        "# # Entropía Normalizada",
        "Máx: 0.678", "",
        "# # Gradiente Promedio", 
        "Máx: 0.042", "",
        "# # Mesa de Simulación",
        "Máx: 0.015 ± 0.002", "",
        "---", "",
        "# # # Módulos",
        "- **Gradiente**: Media=0.505 ± 0.056",
        "- **Fotografía**: Media=0.015 ± 0.002", "",
        "---", "",
        "# # # Modelo",
        "- **Modelo**",
        "- 2.0000",
        "- 4.0000", 
        "- 6.0000",
        "- 8.0000",
        "- 10.0000", "",
        "---", "",
        "# # # Tiempo de Ejecución",
        "Total: 2,32 segundos"
    ]
    
    # Posición inicial y espaciado
    y_position = 0.95
    line_height = 0.05
    
    # Función para determinar el tamaño de fuente
    def get_font_size(line):
        if line.startswith("# # "):
            return 16
        elif line.startswith("# # #"):
            return 14
        else:
            return 12
    
    # Dibujar cada línea del reporte
    for line in report_content:
        if line == "---":
            ax.axhline(y=y_position-0.01, xmin=0.05, xmax=0.95, color='black', linewidth=1)
            y_position -= line_height
            continue
            
        if line:  # Solo dibujar líneas no vacías
            font_size = get_font_size(line)
            ax.text(0.05, y_position, line, 
                   fontsize=font_size, 
                   ha='left', va='top',
                   transform=fig.transFigure)
        y_position -= line_height
    
    # Guardar la imagen
    img_filename = f"BZ_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(img_filename, bbox_inches='tight', dpi=150)
    plt.close()
    
    return img_filename

def generate_timing_file():
    """Genera el archivo de tiempos con formato exacto"""
    # Datos de ejemplo (deberías reemplazarlos con tus tiempos reales)
    timing_data = {
        "Inicialización": 0.0006,
        "Simulación principal": 2.8486,
        "Guardado de datos": 0.0580,
        "Cálculo de entropía": 0.1534,
        "Cálculo de gradiente": 0.1641,
        "Métricas y escritura": 0.3352
    }
    
    # Nombre del archivo con timestamp
    txt_filename = f"BZ_Timings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # Escribir el archivo
    with open(txt_filename, 'w') as f:
        f.write("=== Resultados ===\n")
        f.write("=== Tiempos de ejecución ===\n")
        
        # Escribir cada tiempo con formato consistente
        for key, value in timing_data.items():
            f.write(f"{key}: {value:.4f} s\n")
        
        f.write("---------------------------------\n")
        f.write(f"Datos guardados en: {os.getcwd()}\n")
    
    return txt_filename

def main():
    print("Generando reporte BZ...")
    start_time = time.time()
    
    # Generar la imagen del reporte
    img_file = generate_report_image()
    print(f"Imagen generada: {img_file}")
    
    # Generar el archivo de tiempos
    txt_file = generate_timing_file()
    print(f"Archivo de tiempos generado: {txt_file}")
    
    total_time = time.time() - start_time
    print(f"Proceso completado en {total_time:.2f} segundos")

if __name__ == "__main__":
    main()

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <sys/stat.h>
#include <ctime>
#include <iomanip>
#include <algorithm>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

// ===== CONFIGURACIÓN =====
int N;  // Se define según entrada del usuario
const int steps = 150000;
const int output_interval = 100;
const double dt = 0.1;
const double dx = 1.0;
double dx_sq;  // Se calculará después de conocer N
const double Du = 0.20;
const double Dv = 0.10;
const double F = 0.026;
const double k = 0.053;
const double F_plus_k = F + k;  // Pre-calculado

// Prototipos de funciones
void initialize_BZ(double* u, double* v, int geometry_type, int num_sources, int N);
void create_directory(const string& path);
void save_grid(double* v, int iter, const string& output_dir, int N);
double calculate_normalized_entropy(double* data, int N);
double calculate_average_gradient(double* data, int N);
void print_geometry_options();

// Kernels CUDA
__global__ void initialize_BZ_kernel(double* u, double* v, int geometry_type, int num_sources, int N);
__global__ void bz_simulation_step(double* u, double* v, double* u_next, double* v_next, int N, double dx_sq, double dt);
__global__ void calculate_entropy_kernel(double* data, int* hist, int N, int bins, double bin_size);
__global__ void calculate_gradient_kernel(double* data, double* gradients, int N);

// Función principal
int main() {
    // Variables para medición de tiempos
    double init_time = 0.0;
    double simulation_time = 0.0;
    double save_time = 0.0;
    double entropy_time = 0.0;
    double gradient_time = 0.0;
    double metrics_time = 0.0;
    double total_time = 0.0;
    double memcpy_time = 0.0;

    auto total_start = high_resolution_clock::now();

    cout << "Tamaño de la malla (N x N, recomendado 100-2000): ";
    cin >> N;

    if (N <= 0) {
        cerr << "Error: El tamaño de la malla debe ser positivo." << endl;
        return 1;
    }

    dx_sq = dx * dx;

    print_geometry_options();

    int geometry_type, num_sources;
    cout << "Seleccione el tipo de geometría (1-5): ";
    cin >> geometry_type;

    if (geometry_type < 1 || geometry_type > 5) {
        cerr << "Error: Opción de geometría no válida." << endl;
        return 1;
    }

    if (geometry_type == 1) {
        cout << "Número de focos a crear: ";
        cin >> num_sources;
    } else {
        num_sources = 1;
    }

    string output_dir = "BZ_Geometry_" + to_string(geometry_type);
    create_directory(output_dir);

    // Asignar memoria en el host
    double *u_host = new double[N*N];
    double *v_host = new double[N*N];

    // Asignar memoria en el device
    double *u_device, *v_device, *u_next_device, *v_next_device;
    cudaMalloc(&u_device, N*N*sizeof(double));
    cudaMalloc(&v_device, N*N*sizeof(double));
    cudaMalloc(&u_next_device, N*N*sizeof(double));
    cudaMalloc(&v_next_device, N*N*sizeof(double));

    // Inicialización
    auto init_start = high_resolution_clock::now();
    initialize_BZ(u_host, v_host, geometry_type, num_sources, N);
    
    // Copiar datos al device
    auto memcpy_start = high_resolution_clock::now();
    cudaMemcpy(u_device, u_host, N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(v_device, v_host, N*N*sizeof(double), cudaMemcpyHostToDevice);
    auto memcpy_end = high_resolution_clock::now();
    memcpy_time += duration_cast<duration<double>>(memcpy_end - memcpy_start).count();
    
    auto init_end = high_resolution_clock::now();
    init_time = duration_cast<duration<double>>(init_end - init_start).count();

    cout << "\n=== Simulación Belousov-Zhabotinsky con Geometrías Personalizadas ===\n";
    cout << "Tamaño: " << N << "x" << N << " | Pasos: " << steps << "\n";
    cout << "Geometría seleccionada: ";
    switch(geometry_type) {
        case 1: cout << num_sources << " focos circulares"; break;
        case 2: cout << "Línea horizontal central"; break;
        case 3: cout << "Cuadrado central"; break;
        case 4: cout << "Patrón hexagonal"; break;
        case 5: cout << "Cruz central"; break;
    }
    cout << "\nSalida: " << output_dir << "\n\n";

    // Configuración de métricas
    auto metrics_start = high_resolution_clock::now();
    ofstream metrics(output_dir + "/metrics.csv");
    metrics << "Paso,Entropia,GradientePromedio\n";

    // Métricas iniciales
    auto entropy_start = high_resolution_clock::now();
    double initial_entropy = calculate_normalized_entropy(v_host, N);
    auto entropy_end = high_resolution_clock::now();
    entropy_time += duration_cast<duration<double>>(entropy_end - entropy_start).count();

    auto gradient_start = high_resolution_clock::now();
    double initial_grad = calculate_average_gradient(v_host, N);
    auto gradient_end = high_resolution_clock::now();
    gradient_time += duration_cast<duration<double>>(gradient_end - gradient_start).count();

    metrics << 0 << "," << fixed << setprecision(6) << initial_entropy << "," << initial_grad << "\n";
    cout << "Entropía inicial: " << initial_entropy << "\n";
    auto metrics_end = high_resolution_clock::now();
    metrics_time = duration_cast<duration<double>>(metrics_end - metrics_start).count();

    // Configuración de CUDA
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Simulación principal
    auto sim_start = high_resolution_clock::now();
    for (int n = 1; n <= steps; ++n) {
        bz_simulation_step<<<gridSize, blockSize>>>(u_device, v_device, u_next_device, v_next_device, N, dx_sq, dt);
        cudaDeviceSynchronize();
        
        // Intercambiar punteros para el siguiente paso
        double* temp = u_device;
        u_device = u_next_device;
        u_next_device = temp;
        
        temp = v_device;
        v_device = v_next_device;
        v_next_device = temp;

        if (n % output_interval == 0) {
            auto save_start = high_resolution_clock::now();
            // Copiar v de vuelta al host para guardar y calcular métricas
            cudaMemcpy(v_host, v_device, N*N*sizeof(double), cudaMemcpyDeviceToHost);
            memcpy_time += duration_cast<duration<double>>(high_resolution_clock::now() - save_start).count();
            
            save_grid(v_host, n, output_dir, N);
            auto save_end = high_resolution_clock::now();
            save_time += duration_cast<duration<double>>(save_end - save_start).count();

            auto metrics_step_start = high_resolution_clock::now();
            auto entropy_step_start = high_resolution_clock::now();
            double entropy = calculate_normalized_entropy(v_host, N);
            auto entropy_step_end = high_resolution_clock::now();
            entropy_time += duration_cast<duration<double>>(entropy_step_end - entropy_step_start).count();

            auto gradient_step_start = high_resolution_clock::now();
            double avg_grad = calculate_average_gradient(v_host, N);
            auto gradient_step_end = high_resolution_clock::now();
            gradient_time += duration_cast<duration<double>>(gradient_step_end - gradient_step_start).count();

            metrics << n << "," << fixed << setprecision(6) << entropy << "," << avg_grad << "\n";
            auto metrics_step_end = high_resolution_clock::now();
            metrics_time += duration_cast<duration<double>>(metrics_step_end - metrics_step_start).count();

            cout << "\rProgreso: " << n << "/" << steps
                 << " | Entropía: " << setw(6) << setprecision(3) << entropy
                 << " | ∇: " << setw(6) << avg_grad << flush;
        }
    }
    auto sim_end = high_resolution_clock::now();
    simulation_time = duration_cast<duration<double>>(sim_end - sim_start).count();

    metrics.close();

    // Liberar memoria
    cudaFree(u_device);
    cudaFree(v_device);
    cudaFree(u_next_device);
    cudaFree(v_next_device);
    delete[] u_host;
    delete[] v_host;

    auto total_end = high_resolution_clock::now();
    total_time = duration_cast<duration<double>>(total_end - total_start).count();

    cout << "\n\n=== Resultados ===\n";
    cout << "=== Tiempos de ejecución ===\n";
    cout << "Inicialización: " << fixed << setprecision(4) << init_time << " s\n";
    cout << "Simulación principal: " << simulation_time << " s\n";
    cout << "Transferencia de datos: " << memcpy_time << " s\n";
    cout << "Guardado de datos: " << save_time << " s\n";
    cout << "Cálculo de entropía: " << entropy_time << " s\n";
    cout << "Cálculo de gradiente: " << gradient_time << " s\n";
    cout << "Métricas y escritura: " << metrics_time << " s\n";
    cout << "---------------------------------\n";
    cout << "Suma de tiempos parciales: "
         << (init_time + simulation_time + memcpy_time + save_time + entropy_time + gradient_time + metrics_time)
         << " s\n";
    cout << "Tiempo total medido: " << total_time << " s\n";
    cout << "Datos guardados en:\n";
    cout << "- " << output_dir << "/bz_XXXXX.csv (patrones espaciales)\n";
    cout << "- " << output_dir << "/metrics.csv (métricas cuantitativas)\n";

    return 0;
}

void print_geometry_options() {
    const char* options =
        "================================\n"
        "    Geometrías disponibles:\n"
        "================================\n"
        "1. Focos circulares (especificar número)\n"
        "2. Línea horizontal central\n"
        "3. Cuadrado central\n"
        "4. Patrón hexagonal\n"
        "5. Cruz central\n"
        "================================\n";
    cout << options;
}

void create_directory(const string& path) {
    int status = mkdir(path.c_str(), 0777);
    if (status != 0 && errno != EEXIST) {
        cerr << "Error al crear directorio: " << path << endl;
        exit(1);
    }
}

void initialize_BZ(double* u, double* v, int geometry_type, int num_sources, int N) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    // Inicialización base
    for (int i = 0; i < N*N; ++i) {
        u[i] = 0.8 + 0.05 * dis(gen);
        v[i] = 0.0;
    }

    const double radius = 8.0;
    const double radius_sq = radius * radius;
    const double center = N/2.0;
    const double hex_size = N/5.0;
    const double hex_const = hex_size * 0.866;

    switch(geometry_type) {
        case 1: { // Focos circulares
            const double angle_step = 2.0 * M_PI / num_sources;
            const double dist = N/3.5;

            for (int s = 0; s < num_sources; s++) {
                double angle = angle_step * s;
                double cx = center + dist * cos(angle);
                double cy = center + dist * sin(angle);

                int min_i = max(0, static_cast<int>(cx - radius - 1));
                int max_i = min(N-1, static_cast<int>(cx + radius + 1));
                int min_j = max(0, static_cast<int>(cy - radius - 1));
                int max_j = min(N-1, static_cast<int>(cy + radius + 1));

                for (int i = min_i; i <= max_i; ++i) {
                    double dx = i - cx;
                    for (int j = min_j; j <= max_j; ++j) {
                        double dy = j - cy;
                        if (dx*dx + dy*dy < radius_sq) {
                            v[i*N + j] = 0.9;
                            u[i*N + j] = 0.2;
                        }
                    }
                }
            }
            break;
        }
        case 2: { // Línea horizontal
            int j_start = max(0, static_cast<int>(center-3));
            int j_end = min(N-1, static_cast<int>(center+3));

            for (int i = 0; i < N; ++i) {
                for (int j = j_start; j <= j_end; ++j) {
                    v[i*N + j] = 0.9;
                    u[i*N + j] = 0.2;
                }
            }
            break;
        }
        case 3: { // Cuadrado central
            int size = N/4;
            int i_start = max(0, static_cast<int>(center-size));
            int i_end = min(N-1, static_cast<int>(center+size));
            int j_start = i_start, j_end = i_end;

            for (int i = i_start; i <= i_end; ++i) {
                for (int j = j_start; j <= j_end; ++j) {
                    v[i*N + j] = 0.9;
                    u[i*N + j] = 0.2;
                }
            }
            break;
        }
        case 4: { // Hexágono
            int i_start = max(0, static_cast<int>(center-hex_size));
            int i_end = min(N-1, static_cast<int>(center+hex_size));
            int j_start = max(0, static_cast<int>(center-hex_const));
            int j_end = min(N-1, static_cast<int>(center+hex_const));

            for (int i = i_start; i <= i_end; ++i) {
                double dx_val = abs(i - center);
                for (int j = j_start; j <= j_end; ++j) {
                    double dy_val = abs(j - center);
                    if (dx_val <= hex_size && dy_val <= hex_const &&
                        (0.5*hex_size + 0.866*dy_val) <= hex_size) {
                        v[i*N + j] = 0.9;
                        u[i*N + j] = 0.2;
                    }
                }
            }
            break;
        }
        case 5: { // Cruz
            int center_start = max(0, static_cast<int>(center-2));
            int center_end = min(N-1, static_cast<int>(center+2));

            // Parte horizontal
            for (int i = 0; i < N; ++i) {
                for (int j = center_start; j <= center_end; ++j) {
                    v[i*N + j] = 0.9;
                    u[i*N + j] = 0.2;
                }
            }
            // Parte vertical
            for (int j = 0; j < N; ++j) {
                for (int i = center_start; i <= center_end; ++i) {
                    v[i*N + j] = 0.9;
                    u[i*N + j] = 0.2;
                }
            }
            break;
        }
    }

    // Pequeña perturbación en el resto de la matriz v
    for (int i = 0; i < N*N; ++i) {
        if (v[i] == 0.0) v[i] = 0.001 * dis(gen);
    }
}

__global__ void bz_simulation_step(double* u, double* v, double* u_next, double* v_next, int N, double dx_sq, double dt) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= N || j >= N) return;
    
    int idx = i * N + j;
    double u_val = u[idx];
    double v_val = v[idx];
    double uvv = u_val * v_val * v_val;

    // Cálculo del laplaciano con condiciones de contorno periódicas
    double lap_u = (u[((i+1)%N)*N + j] + u[((i-1+N)%N)*N + j] +
                   u[i*N + (j+1)%N] + u[i*N + (j-1+N)%N] - 4.0 * u_val) / dx_sq;
    
    double lap_v = (v[((i+1)%N)*N + j] + v[((i-1+N)%N)*N + j] +
                   v[i*N + (j+1)%N] + v[i*N + (j-1+N)%N] - 4.0 * v_val) / dx_sq;

    u_next[idx] = max(0.0, min(1.5, u_val + dt * (Du * lap_u - uvv + F * (1.0 - u_val)));
    v_next[idx] = max(0.0, min(1.0, v_val + dt * (Dv * lap_v + uvv - F_plus_k * v_val)));
}

void save_grid(double* v, int iter, const string& output_dir, int N) {
    string filename = output_dir + "/bz_" + to_string(iter) + ".csv";
    ofstream out(filename);

    if (!out.is_open()) {
        cerr << "Error al crear archivo: " << filename << endl;
        return;
    }

    out << fixed << setprecision(6);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            out << v[i*N + j];
            if (j != N-1) out << ",";
        }
        out << "\n";
    }
    out.close();
}

double calculate_normalized_entropy(double* data, int N) {
    const int bins = 20;
    const double bin_size = 1.0 / bins;
    const double log_bins = log(bins);
    vector<int> hist(bins, 0);
    const double total = N*N;
    const double inv_total = 1.0 / total;

    for (int i = 0; i < N*N; ++i) {
        int bin = min(bins-1, static_cast<int>(data[i] / bin_size));
        hist[bin]++;
    }

    double entropy = 0.0;
    for (int count : hist) {
        if (count > 0) {
            double p = count * inv_total;
            entropy -= p * log(p);
        }
    }
    return entropy / log_bins;
}

double calculate_average_gradient(double* data, int N) {
    double total_gradient = 0.0;
    int gradient_count = 0;
    const int N_minus_1 = N - 1;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N_minus_1; ++j) {
            total_gradient += abs(data[i*N + j+1] - data[i*N + j]);
            gradient_count++;
        }
    }

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N_minus_1; ++i) {
            total_gradient += abs(data[(i+1)*N + j] - data[i*N + j]);
            gradient_count++;
        }
    }

    return gradient_count ? total_gradient / gradient_count : 0.0;
}

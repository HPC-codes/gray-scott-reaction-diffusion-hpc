#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <sys/stat.h>
#include <errno.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cuda_profiler_api.h>

using namespace std;
using namespace std::chrono;

// Macro para verificación de errores CUDA
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
             << cudaGetErrorString(err) << endl; \
        exit(EXIT_FAILURE); \
    } \
}

// ===== CONFIGURACIÓN =====
int N;  // Tamaño de la malla (N x N)
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

// Sistema de escritura asíncrona mejorado
const int MAX_SAVE_THREADS = 4;  // Número máximo de hilos de guardado

struct SaveTask {
    double* d_data;
    string filename;
    int step;
};

queue<SaveTask> save_queues[MAX_SAVE_THREADS];
mutex queue_mutexes[MAX_SAVE_THREADS];
condition_variable queue_cvs[MAX_SAVE_THREADS];
atomic<bool> writers_running[MAX_SAVE_THREADS];
vector<thread> writer_threads;

// Functor para calcular valor absoluto
struct absolute_value_functor {
    __device__ double operator()(double x) const {
        return fabs(x);
    }
};

// Writer thread function mejorada para archivos binarios
void writer_thread_function(int thread_id, const string& output_dir) {
    vector<double> host_buffer(N*N);

    while (writers_running[thread_id] || !save_queues[thread_id].empty()) {
        unique_lock<mutex> lock(queue_mutexes[thread_id]);

        queue_cvs[thread_id].wait(lock, [thread_id] {
            return !save_queues[thread_id].empty() || !writers_running[thread_id];
        });

        if (!save_queues[thread_id].empty()) {
            SaveTask task = move(save_queues[thread_id].front());
            save_queues[thread_id].pop();
            lock.unlock();

            // Copia asíncrona con un stream dedicado
            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));
            CUDA_CHECK(cudaMemcpyAsync(host_buffer.data(), task.d_data, N*N*sizeof(double),
                          cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaStreamDestroy(stream));

            // Escribir archivo binario
            string full_path = output_dir + "/bz_" + to_string(task.step) + ".bin";
            ofstream out(full_path, ios::binary);
            if (out.is_open()) {
                // Escribir dimensiones (N, N) y luego los datos
                out.write(reinterpret_cast<const char*>(&N), sizeof(int));
                out.write(reinterpret_cast<const char*>(&N), sizeof(int));
                out.write(reinterpret_cast<const char*>(host_buffer.data()), N*N*sizeof(double));
                if (!out) {
                    cerr << "Error al escribir archivo: " << full_path << endl;
                }
                out.close();
            } else {
                cerr << "Error al abrir archivo: " << full_path << endl;
            }

            // Liberar memoria del dispositivo
            CUDA_CHECK(cudaFree(task.d_data));
        }
    }
}

// Kernels CUDA
__global__ void combined_bz_kernel(double* u, double* v, double* u_next, double* v_next,
                                 int N, double dt, double inv_dx_sq, double Du, double Dv,
                                 double F, double F_plus_k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N) {
        int idx = i * N + j;

        // Calcular laplaciano (5-point stencil)
        double lap_u = (u[((i+1)%N)*N + j] + u[((i-1+N)%N)*N + j] +
                      u[i*N + (j+1)%N] + u[i*N + (j-1+N)%N] -
                      4.0 * u[idx]) * inv_dx_sq;

        double lap_v = (v[((i+1)%N)*N + j] + v[((i-1+N)%N)*N + j] +
                      v[i*N + (j+1)%N] + v[i*N + (j-1+N)%N] -
                      4.0 * v[idx]) * inv_dx_sq;

        // Reacciones químicas
        double u_val = u[idx];
        double v_val = v[idx];
        double uvv = u_val * v_val * v_val;

        // Actualizar con condiciones de frontera periódicas
        u_next[idx] = max(0.0, min(1.5, u_val + dt * (Du * lap_u - uvv + F * (1.0 - u_val))));
        v_next[idx] = max(0.0, min(1.0, v_val + dt * (Dv * lap_v + uvv - F_plus_k * v_val)));
    }
}

__global__ void entropy_histogram_kernel(const double* data, int* hist, int N, int bins, double bin_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N) {
        double val = data[i * N + j];
        int bin = min(bins-1, static_cast<int>(val / bin_size));
        atomicAdd(&hist[bin], 1);
    }
}

__global__ void calculate_gradients_kernel(const double* v, double* grad_x, double* grad_y, int N, double inv_2dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < N-1 && j > 0 && j < N-1) {
        int idx = i * N + j;
        grad_x[idx] = (v[(i+1)*N + j] - v[(i-1)*N + j]) * inv_2dx;
        grad_y[idx] = (v[i*N + (j+1)] - v[i*N + (j-1)]) * inv_2dx;
    } else if (i < N && j < N) {
        // Condiciones de frontera periódicas
        int idx = i * N + j;
        grad_x[idx] = (v[((i+1)%N)*N + j] - v[((i-1+N)%N)*N + j]) * inv_2dx;
        grad_y[idx] = (v[i*N + (j+1)%N] - v[i*N + (j-1+N)%N]) * inv_2dx;
    }
}

// Funciones auxiliares
void initialize_BZ_cuda(double* d_u, double* d_v, int N, int geometry_type, int num_sources) {
    vector<double> u(N*N, 0.8);
    vector<double> v(N*N, 0.0);

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 0.05);

    for (auto& val : u) {
        val += dis(gen);
    }

    const double radius = 8.0;
    const double radius_sq = radius * radius;
    const double center = N/2.0;
    const double hex_size = N/5.0;
    const double hex_const = hex_size * 0.866;

    switch(geometry_type) {
        case 1: {
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
        case 2: {
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
        case 3: {
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
        case 4: {
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
        case 5: {
            int center_start = max(0, static_cast<int>(center-2));
            int center_end = min(N-1, static_cast<int>(center+2));

            for (int i = 0; i < N; ++i) {
                for (int j = center_start; j <= center_end; ++j) {
                    v[i*N + j] = 0.9;
                    u[i*N + j] = 0.2;
                }
            }
            for (int j = 0; j < N; ++j) {
                for (int i = center_start; i <= center_end; ++i) {
                    v[i*N + j] = 0.9;
                    u[i*N + j] = 0.2;
                }
            }
            break;
        }
    }

    uniform_real_distribution<> dis_v(0.0, 0.001);
    for (auto& val : v) {
        if (val == 0.0) val = dis_v(gen);
    }

    CUDA_CHECK(cudaMemcpy(d_u, u.data(), N*N*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, v.data(), N*N*sizeof(double), cudaMemcpyHostToDevice));
}

void save_grid_binary_optimized(double* d_v, int step, const string& output_dir) {
    double* d_v_copy;
    CUDA_CHECK(cudaMalloc(&d_v_copy, N*N*sizeof(double)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemcpyAsync(d_v_copy, d_v, N*N*sizeof(double), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    static atomic<int> next_queue(0);
    int queue_id = next_queue++ % MAX_SAVE_THREADS;

    SaveTask task;
    task.d_data = d_v_copy;
    task.filename = output_dir + "/bz_" + to_string(step) + ".bin";
    task.step = step;

    lock_guard<mutex> lock(queue_mutexes[queue_id]);
    save_queues[queue_id].push(move(task));
    queue_cvs[queue_id].notify_one();
}

double calculate_entropy_thrust(double* d_v, int N) {
    const int bins = 20;
    const double bin_size = 1.0 / bins;
    const double log_bins = log(bins);
    int* d_hist;

    CUDA_CHECK(cudaMalloc(&d_hist, bins * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_hist, 0, bins * sizeof(int)));

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    entropy_histogram_kernel<<<gridSize, blockSize>>>(d_v, d_hist, N, bins, bin_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<int> hist(bins);
    CUDA_CHECK(cudaMemcpy(hist.data(), d_hist, bins * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_hist));

    double entropy = 0.0;
    const double total = N*N;
    const double inv_total = 1.0 / total;

    for (int count : hist) {
        if (count > 0) {
            double p = count * inv_total;
            entropy -= p * log(p);
        }
    }
    return entropy / log_bins;
}

double calculate_average_gradient_cuda(double* d_v, int N) {
    double* d_grad_x, *d_grad_y;
    CUDA_CHECK(cudaMalloc(&d_grad_x, N*N*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grad_y, N*N*sizeof(double)));

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    double inv_2dx = 1.0 / (2.0 * dx);
    calculate_gradients_kernel<<<gridSize, blockSize>>>(d_v, d_grad_x, d_grad_y, N, inv_2dx);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    thrust::device_ptr<double> dev_ptr_x(d_grad_x);
    thrust::device_ptr<double> dev_ptr_y(d_grad_y);

    double sum_x = thrust::transform_reduce(dev_ptr_x, dev_ptr_x + N*N,
        absolute_value_functor(), 0.0, thrust::plus<double>());
    double sum_y = thrust::transform_reduce(dev_ptr_y, dev_ptr_y + N*N,
        absolute_value_functor(), 0.0, thrust::plus<double>());

    CUDA_CHECK(cudaFree(d_grad_x));
    CUDA_CHECK(cudaFree(d_grad_y));

    return (sum_x + sum_y) / (2.0 * N * N);
}

void print_geometry_options() {
    cout << "================================\n"
         << "    Geometrías disponibles:\n"
         << "================================\n"
         << "1. Focos circulares (especificar número)\n"
         << "2. Línea horizontal central\n"
         << "3. Cuadrado central\n"
         << "4. Patrón hexagonal\n"
         << "5. Cruz central\n"
         << "================================\n";
}

void create_directory(const string& path) {
    int status = mkdir(path.c_str(), 0777);
    if (status != 0 && errno != EEXIST) {
        cerr << "Error al crear directorio: " << path << endl;
        exit(1);
    }
}

int main() {
    // Configurar dispositivo CUDA
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        cerr << "No se encontraron dispositivos CUDA" << endl;
        return 1;
    }

    int device_id = 0;
    CUDA_CHECK(cudaSetDevice(device_id));

    cudaDeviceProp device_prop;
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
    cout << "Usando dispositivo CUDA: " << device_prop.name << endl;

    cudaProfilerStart();

    // Variables para medición de tiempos
    double init_time = 0.0;
    double simulation_time = 0.0;
    double save_time = 0.0;
    double entropy_time = 0.0;
    double gradient_time = 0.0;
    double metrics_time = 0.0;
    double total_time = 0.0;

    auto total_start = high_resolution_clock::now();

    cout << "Tamaño de la malla (N x N, recomendado 100-2000): ";
    cin >> N;

    if (N <= 0) {
        cerr << "Error: El tamaño de la malla debe ser positivo." << endl;
        return 1;
    }

    dx_sq = dx * dx;
    double inv_dx_sq = 1.0 / dx_sq;

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

    // Iniciar los hilos de escritura
    for (int i = 0; i < MAX_SAVE_THREADS; ++i) {
        writers_running[i] = true;
        writer_threads.emplace_back(writer_thread_function, i, output_dir);
    }

    // Configuración CUDA
    size_t grid_size = N * N * sizeof(double);
    double *d_u, *d_v, *d_u_next, *d_v_next;

    CUDA_CHECK(cudaMalloc(&d_u, grid_size));
    CUDA_CHECK(cudaMalloc(&d_v, grid_size));
    CUDA_CHECK(cudaMalloc(&d_u_next, grid_size));
    CUDA_CHECK(cudaMalloc(&d_v_next, grid_size));

    // Inicialización en GPU
    auto init_start = high_resolution_clock::now();
    initialize_BZ_cuda(d_u, d_v, N, geometry_type, num_sources);
    CUDA_CHECK(cudaMemset(d_u_next, 0, grid_size));
    CUDA_CHECK(cudaMemset(d_v_next, 0, grid_size));
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

    // Configuración de métricas (archivo binario)
    auto metrics_start = high_resolution_clock::now();
    ofstream metrics(output_dir + "/metrics.bin", ios::binary | ios::trunc);

    if (!metrics.is_open()) {
        cerr << "Error: No se pudo abrir el archivo de métricas para escritura" << endl;
        return 1;
    }

    // Métricas iniciales
    auto entropy_start = high_resolution_clock::now();
    double initial_entropy = calculate_entropy_thrust(d_v, N);
    auto entropy_end = high_resolution_clock::now();
    entropy_time += duration_cast<duration<double>>(entropy_end - entropy_start).count();

    auto gradient_start = high_resolution_clock::now();
    double initial_grad = calculate_average_gradient_cuda(d_v, N);
    auto gradient_end = high_resolution_clock::now();
    gradient_time += duration_cast<duration<double>>(gradient_end - gradient_start).count();

    // Escribir métricas iniciales en binario
    int step_zero = 0;
    metrics.write(reinterpret_cast<const char*>(&step_zero), sizeof(int));
    metrics.write(reinterpret_cast<const char*>(&initial_entropy), sizeof(double));
    metrics.write(reinterpret_cast<const char*>(&initial_grad), sizeof(double));

    if (!metrics) {
        cerr << "Error al escribir métricas iniciales" << endl;
        return 1;
    }

    cout << "Entropía inicial: " << initial_entropy << "\n";
    auto metrics_end = high_resolution_clock::now();
    metrics_time = duration_cast<duration<double>>(metrics_end - metrics_start).count();

    // Configuración de kernels
    dim3 blockSize(32, 32);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Simulación principal
    auto sim_start = high_resolution_clock::now();
    for (int n = 1; n <= steps; ++n) {
        combined_bz_kernel<<<gridSize, blockSize>>>(d_u, d_v, d_u_next, d_v_next,
                                                  N, dt, inv_dx_sq, Du, Dv, F, F_plus_k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        swap(d_u, d_u_next);
        swap(d_v, d_v_next);

        if (n % output_interval == 0) {
            auto save_start = high_resolution_clock::now();
            save_grid_binary_optimized(d_v, n, output_dir);
            auto save_end = high_resolution_clock::now();
            save_time += duration_cast<duration<double>>(save_end - save_start).count();

            auto metrics_step_start = high_resolution_clock::now();
            auto entropy_step_start = high_resolution_clock::now();
            double entropy = calculate_entropy_thrust(d_v, N);
            auto entropy_step_end = high_resolution_clock::now();
            entropy_time += duration_cast<duration<double>>(entropy_step_end - entropy_step_start).count();

            auto gradient_step_start = high_resolution_clock::now();
            double avg_grad = calculate_average_gradient_cuda(d_v, N);
            auto gradient_step_end = high_resolution_clock::now();
            gradient_time += duration_cast<duration<double>>(gradient_step_end - gradient_step_start).count();

            // Escribir métricas en binario con verificación
            metrics.write(reinterpret_cast<const char*>(&n), sizeof(int));
            metrics.write(reinterpret_cast<const char*>(&entropy), sizeof(double));
            metrics.write(reinterpret_cast<const char*>(&avg_grad), sizeof(double));

            if (!metrics) {
                cerr << "\nError al escribir métricas en el paso " << n << endl;
                break;
            }

            cout << "\rProgreso: " << n << "/" << steps
                 << " | Entropía: " << setw(6) << setprecision(3) << entropy
                 << " | ∇: " << setw(6) << avg_grad << flush;

            auto metrics_step_end = high_resolution_clock::now();
            metrics_time += duration_cast<duration<double>>(metrics_step_end - metrics_step_start).count();
        }
    }
    auto sim_end = high_resolution_clock::now();
    simulation_time = duration_cast<duration<double>>(sim_end - sim_start).count();

    // Asegurar que todos los datos se escriban
    metrics.flush();
    if (!metrics) {
        cerr << "Error al flush del archivo de métricas" << endl;
    }
    metrics.close();

    // Detener los hilos de escritura
    for (int i = 0; i < MAX_SAVE_THREADS; ++i) {
        writers_running[i] = false;
        queue_cvs[i].notify_one();
    }

    for (auto& thread : writer_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // Liberar memoria GPU
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_u_next));
    CUDA_CHECK(cudaFree(d_v_next));

    auto total_end = high_resolution_clock::now();
    total_time = duration_cast<duration<double>>(total_end - total_start).count();

    cout << "\n\n=== Resultados ===\n";
    cout << "=== Tiempos de ejecución ===\n";
    cout << "Inicialización: " << fixed << setprecision(4) << init_time << " s\n";
    cout << "Simulación principal: " << simulation_time << " s\n";
    cout << "Guardado de datos: " << save_time << " s\n";
    cout << "Cálculo de entropía: " << entropy_time << " s\n";
    cout << "Cálculo de gradiente: " << gradient_time << " s\n";
    cout << "Métricas y escritura: " << metrics_time << " s\n";
    cout << "---------------------------------\n";
    cout << "Suma de tiempos parciales: "
         << (init_time + simulation_time + save_time + entropy_time + gradient_time + metrics_time)
         << " s\n";
    cout << "Tiempo total medido: " << total_time << " s\n";
    cout << "Datos guardados en:\n";
    cout << "- " << output_dir << "/bz_XXXXX.bin (patrones espaciales en binario)\n";
    cout << "- " << output_dir << "/metrics.bin (métricas cuantitativas en binario)\n";

    cudaProfilerStop();
    return 0;
}
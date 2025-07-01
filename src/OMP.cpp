#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <sys/stat.h>
#include <algorithm>
#include <omp.h>
#include <atomic>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

using namespace std;
using namespace std::chrono;

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
const double F_plus_k = F + k;

// Sistema de escritura asíncrona
const int MAX_SAVE_THREADS = 4;
struct SaveTask {
    vector<double> data;
    string filename;
    int step;
};

queue<SaveTask> save_queues[MAX_SAVE_THREADS];
mutex queue_mutexes[MAX_SAVE_THREADS];
condition_variable queue_cvs[MAX_SAVE_THREADS];
atomic<bool> writers_running[MAX_SAVE_THREADS];
vector<thread> writer_threads;

// Función clamp para evitar conflictos
template<typename T>
const T& my_clamp(const T& v, const T& lo, const T& hi) {
    return (v < lo) ? lo : (hi < v) ? hi : v;
}

// Función para calcular el laplaciano optimizado
inline double laplacian(const vector<double>& grid, int i, int j, int N, double inv_dx_sq) {
    const int im1 = (i == 0) ? N-1 : i-1;
    const int ip1 = (i == N-1) ? 0 : i+1;
    const int jm1 = (j == 0) ? N-1 : j-1;
    const int jp1 = (j == N-1) ? 0 : j+1;

    return (grid[ip1*N + j] + grid[im1*N + j] + 
           grid[i*N + jp1] + grid[i*N + jm1] - 
           4.0 * grid[i*N + j]) * inv_dx_sq;
}

// Writer thread function optimizada
void writer_thread_function(int thread_id, const string& output_dir) {
    while (writers_running[thread_id] || !save_queues[thread_id].empty()) {
        unique_lock<mutex> lock(queue_mutexes[thread_id]);
        queue_cvs[thread_id].wait(lock, [thread_id] {
            return !save_queues[thread_id].empty() || !writers_running[thread_id];
        });

        if (!save_queues[thread_id].empty()) {
            SaveTask task = move(save_queues[thread_id].front());
            save_queues[thread_id].pop();
            lock.unlock();

            ofstream out(task.filename, ios::binary);
            if (out.is_open()) {
                out.write(reinterpret_cast<const char*>(&N), sizeof(int));
                out.write(reinterpret_cast<const char*>(&N), sizeof(int));
                out.write(reinterpret_cast<const char*>(task.data.data()), N*N*sizeof(double));
                out.close();
            }
        }
    }
}

// Inicialización completa con todas las geometrías
void initialize_BZ(vector<double>& u, vector<double>& v, int N, int geometry_type, int num_sources) {
    random_device rd;
    vector<mt19937> gens(omp_get_max_threads());
    for (auto& gen : gens) gen.seed(rd() + omp_get_thread_num());

    #pragma omp parallel
    {
        uniform_real_distribution<> dis(0.0, 0.05);
        int thread_num = omp_get_thread_num();
        
        #pragma omp for
        for (int i = 0; i < N*N; ++i) {
            u[i] = 0.8 + dis(gens[thread_num]);
            v[i] = 0.001 * dis(gens[thread_num]);
        }
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

            #pragma omp parallel for
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

            #pragma omp parallel for
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

            #pragma omp parallel for
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

            #pragma omp parallel for
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
            #pragma omp parallel for
            for (int i = 0; i < N; ++i) {
                for (int j = center_start; j <= center_end; ++j) {
                    v[i*N + j] = 0.9;
                    u[i*N + j] = 0.2;
                }
            }
            // Parte vertical
            #pragma omp parallel for
            for (int j = 0; j < N; ++j) {
                for (int i = center_start; i <= center_end; ++i) {
                    v[i*N + j] = 0.9;
                    u[i*N + j] = 0.2;
                }
            }
            break;
        }
    }
}

// Guardado paralelo optimizado
void save_grid_parallel(const vector<double>& v, int step, const string& output_dir) {
    static atomic<int> next_queue(0);
    int queue_id = next_queue++ % MAX_SAVE_THREADS;

    SaveTask task;
    task.data = v; // Copia los datos
    task.filename = output_dir + "/bz_" + to_string(step) + ".bin";
    task.step = step;

    lock_guard<mutex> lock(queue_mutexes[queue_id]);
    save_queues[queue_id].push(move(task));
    queue_cvs[queue_id].notify_one();
}

// Cálculo de entropía completo
double calculate_entropy(const vector<double>& v, int N) {
    constexpr int bins = 20;
    constexpr double bin_size = 1.0 / bins;
    constexpr double log_bins = log(bins);
    vector<int> hist(bins, 0);

    #pragma omp parallel
    {
        vector<int> local_hist(bins, 0);
        
        #pragma omp for nowait
        for (int i = 0; i < N*N; ++i) {
            int bin = min(bins-1, static_cast<int>(v[i] / bin_size));
            local_hist[bin]++;
        }
        
        #pragma omp critical
        {
            for (int b = 0; b < bins; ++b) {
                hist[b] += local_hist[b];
            }
        }
    }

    double entropy = 0.0;
    const double inv_total = 1.0 / (N*N);

    #pragma omp parallel for reduction(+:entropy)
    for (int count : hist) {
        if (count > 0) {
            double p = count * inv_total;
            entropy -= p * log(p);
        }
    }
    return entropy / log_bins;
}

// Cálculo de gradiente completo
double calculate_average_gradient(const vector<double>& v, int N) {
    double total_grad = 0.0;
    const double inv_2dx = 1.0 / (2.0 * dx);

    #pragma omp parallel for reduction(+:total_grad)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int im1 = (i == 0) ? N-1 : i-1;
            int ip1 = (i == N-1) ? 0 : i+1;
            int jm1 = (j == 0) ? N-1 : j-1;
            int jp1 = (j == N-1) ? 0 : j+1;

            double grad_x = (v[ip1*N + j] - v[im1*N + j]) * inv_2dx;
            double grad_y = (v[i*N + jp1] - v[i*N + jm1]) * inv_2dx;
            total_grad += (fabs(grad_x) + fabs(grad_y));
        }
    }
    return total_grad / (2.0 * N * N);
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

    // Iniciar hilos de escritura
    for (int i = 0; i < MAX_SAVE_THREADS; ++i) {
        writers_running[i] = true;
        writer_threads.emplace_back(writer_thread_function, i, output_dir);
    }

    // Configuración de grids
    vector<double> u(N*N), v(N*N), u_next(N*N), v_next(N*N);

    // Inicialización
    auto start = high_resolution_clock::now();
    initialize_BZ(u, v, N, geometry_type, num_sources);

    // Configurar OpenMP
    omp_set_num_threads(omp_get_max_threads());
    omp_set_schedule(omp_sched_guided, 64/sizeof(double));

    // Archivo de métricas binario
    ofstream metrics(output_dir + "/metrics.bin", ios::binary);
    if (!metrics.is_open()) {
        cerr << "Error al crear archivo de métricas" << endl;
        return 1;
    }

    // Métricas iniciales
    double initial_entropy = calculate_entropy(v, N);
    double initial_grad = calculate_average_gradient(v, N);
    
    // Escribir métricas iniciales
    int step_zero = 0;
    metrics.write(reinterpret_cast<const char*>(&step_zero), sizeof(int));
    metrics.write(reinterpret_cast<const char*>(&initial_entropy), sizeof(double));
    metrics.write(reinterpret_cast<const char*>(&initial_grad), sizeof(double));

    cout << "\n=== Simulación Belousov-Zhabotinsky ===\n";
    cout << "Tamaño: " << N << "x" << N << " | Pasos: " << steps << "\n";
    cout << "Geometría: " << geometry_type << " | Focos: " << num_sources << "\n";
    cout << "Entropía inicial: " << initial_entropy << "\n";

    // Simulación principal
    for (int n = 1; n <= steps; ++n) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                int idx = i*N + j;
                double u_val = u[idx];
                double v_val = v[idx];
                double uvv = u_val * v_val * v_val;
                
                double lap_u = laplacian(u, i, j, N, inv_dx_sq);
                double lap_v = laplacian(v, i, j, N, inv_dx_sq);
                
                u_next[idx] = my_clamp(u_val + dt * (Du * lap_u - uvv + F * (1.0 - u_val)), 0.0, 1.5);
                v_next[idx] = my_clamp(v_val + dt * (Dv * lap_v + uvv - F_plus_k * v_val), 0.0, 1.0);
            }
        }
        swap(u, u_next);
        swap(v, v_next);

        if (n % output_interval == 0) {
            save_grid_parallel(v, n, output_dir);

            double entropy = calculate_entropy(v, N);
            double avg_grad = calculate_average_gradient(v, N);
            
            // Escribir métricas
            metrics.write(reinterpret_cast<const char*>(&n), sizeof(int));
            metrics.write(reinterpret_cast<const char*>(&entropy), sizeof(double));
            metrics.write(reinterpret_cast<const char*>(&avg_grad), sizeof(double));

            cout << "\rProgreso: " << n << "/" << steps
                 << " | Entropía: " << fixed << setprecision(4) << entropy
                 << " | ∇: " << avg_grad << flush;
        }
    }

    // Finalización
    metrics.close();
    
    for (int i = 0; i < MAX_SAVE_THREADS; ++i) {
        writers_running[i] = false;
        queue_cvs[i].notify_one();
    }

    for (auto& t : writer_threads) {
        if (t.joinable()) t.join();
    }

    auto end = high_resolution_clock::now();
    double total_time = duration_cast<duration<double>>(end - start).count();

    cout << "\n\n=== Simulación completada ===";
    cout << "\nTiempo total: " << total_time << " s";
    cout << "\nDatos guardados en: " << output_dir << "/\n";

    return 0;
}


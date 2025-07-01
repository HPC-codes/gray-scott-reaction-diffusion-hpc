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
#include <omp.h>
#include <cmath>      // Para log

using namespace std;
using namespace std::chrono;

// ===== CONFIGURACIÓN =====
int N;  // Ahora se define según entrada del usuario
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

using Grid = vector<vector<double>>;

// Prototipos
void initialize_BZ(Grid& u, Grid& v, int geometry_type, int num_sources);
inline double laplacian(const Grid& g, int i, int j);
void create_directory(const string& path);
void save_grid(const Grid& v, int iter, const string& output_dir);
double calculate_normalized_entropy(const Grid& data);
double calculate_average_gradient(const Grid& data);
void print_geometry_options();
void run_simulation(int num_threads, int N, int geometry_type, int num_sources, vector<double>& times);

int main() {
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

    // Vector para almacenar tiempos de ejecución para diferentes números de hilos
    vector<double> execution_times;

    // Ejecutar la simulación con diferentes números de hilos (de 2 a 8)
    for (int num_threads = 2; num_threads <= 8; num_threads++) {
        run_simulation(num_threads, N, geometry_type, num_sources, execution_times);
    }

    // Guardar los tiempos de ejecución para generar la gráfica
    ofstream time_data("threads_vs_time.csv");
    time_data << "Hilos,Tiempo\n";
    for (size_t i = 0; i < execution_times.size(); ++i) {
        time_data << (i+2) << "," << fixed << setprecision(4) << execution_times[i] << "\n";
    }
    time_data.close();

    cout << "\nDatos de tiempo vs hilos guardados en: threads_vs_time.csv\n";
    cout << "Puede usar este archivo para generar una gráfica de tiempo vs hilos.\n";

    return 0;
}

void run_simulation(int num_threads, int N, int geometry_type, int num_sources, vector<double>& times) {
    double total_time = 0.0;
    auto total_start = high_resolution_clock::now();

    // Configurar número de hilos
    omp_set_num_threads(num_threads);

    string output_dir = "BZ_Geometry_" + to_string(geometry_type) + "_threads_" + to_string(num_threads);
    create_directory(output_dir);

    // Inicialización
    Grid u, v;
    u.resize(N, vector<double>(N, 1.0));
    v.resize(N, vector<double>(N, 0.0));

    auto init_start = high_resolution_clock::now();
    initialize_BZ(u, v, geometry_type, num_sources);
    auto init_end = high_resolution_clock::now();
    double init_time = duration_cast<duration<double>>(init_end - init_start).count();

    cout << "\n=== Simulación con " << num_threads << " hilos ===\n";
    cout << "Tamaño: " << N << "x" << N << " | Pasos: " << steps << "\n";

    // Configuración de métricas
    ofstream metrics(output_dir + "/metrics.csv");
    metrics << "Paso,Entropia,GradientePromedio\n";

    // Métricas iniciales
    double initial_entropy = calculate_normalized_entropy(v);
    double initial_grad = calculate_average_gradient(v);
    metrics << 0 << "," << fixed << setprecision(6) << initial_entropy << "," << initial_grad << "\n";

    // Simulación principal
    auto sim_start = high_resolution_clock::now();
    for (int n = 1; n <= steps; ++n) {
        Grid u_next = u;
        Grid v_next = v;

        // Paralelización del bucle principal de cálculo
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double u_val = u[i][j];
                double v_val = v[i][j];
                double uvv = u_val * v_val * v_val;

                double lap_u = laplacian(u, i, j);
                double lap_v = laplacian(v, i, j);

                u_next[i][j] = std::max(0.0, std::min(1.5, u_val + dt * (Du * lap_u - uvv + F * (1.0 - u_val))));
                v_next[i][j] = max(0.0, min(1.0, v_val + dt * (Dv * lap_v + uvv - F_plus_k * v_val)));
            }
        }
        u = move(u_next);
        v = move(v_next);

        if (n % output_interval == 0) {
            save_grid(v, n, output_dir);

            double entropy = calculate_normalized_entropy(v);
            double avg_grad = calculate_average_gradient(v);
            metrics << n << "," << fixed << setprecision(6) << entropy << "," << avg_grad << "\n";

            cout << "\rProgreso: " << n << "/" << steps
                 << " | Entropía: " << setw(6) << setprecision(3) << entropy
                 << " | ∇: " << setw(6) << avg_grad << flush;
        }
    }
    auto sim_end = high_resolution_clock::now();
    double simulation_time = duration_cast<duration<double>>(sim_end - sim_start).count();

    metrics.close();

    auto total_end = high_resolution_clock::now();
    total_time = duration_cast<duration<double>>(total_end - total_start).count();

    cout << "\nTiempo total con " << num_threads << " hilos: " << fixed << setprecision(4) << total_time << " s\n";
    times.push_back(total_time);
}

// Las demás funciones permanecen igual que en el código original
// [Aquí irían todas las demás funciones sin cambios: print_geometry_options, create_directory, 
// initialize_BZ, laplacian, save_grid, calculate_normalized_entropy, calculate_average_gradient]

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

void initialize_BZ(Grid& u, Grid& v, int geometry_type, int num_sources) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    // Inicialización base
    #pragma omp parallel for
    for (auto& row : u) {
        for (auto& val : row) {
            val = 0.8 + 0.05 * dis(gen);
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
                            v[i][j] = 0.9;
                            u[i][j] = 0.2;
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
                    v[i][j] = 0.9;
                    u[i][j] = 0.2;
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
                    v[i][j] = 0.9;
                    u[i][j] = 0.2;
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
                        v[i][j] = 0.9;
                        u[i][j] = 0.2;
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
                    v[i][j] = 0.9;
                    u[i][j] = 0.2;
                }
            }
            // Parte vertical
            #pragma omp parallel for
            for (int j = 0; j < N; ++j) {
                for (int i = center_start; i <= center_end; ++i) {
                    v[i][j] = 0.9;
                    u[i][j] = 0.2;
                }
            }
            break;
        }
    }

    // Pequeña perturbación en el resto de la matriz v
    #pragma omp parallel for
    for (auto& row : v) {
        for (auto& val : row) {
            if (val == 0.0) val = 0.001 * dis(gen);
        }
    }
}

inline double laplacian(const Grid& g, int i, int j) {
    return (g[(i+1)%N][j] + g[(i-1+N)%N][j] +
           g[i][(j+1)%N] + g[i][(j-1+N)%N] -
           4.0 * g[i][j]) / dx_sq;
}

void save_grid(const Grid& v, int iter, const string& output_dir) {
    string filename = output_dir + "/bz_" + to_string(iter) + ".csv";
    ofstream out(filename);

    if (!out.is_open()) {
        cerr << "Error al crear archivo: " << filename << endl;
        return;
    }

    out << fixed << setprecision(6);
    for (const auto& row : v) {
        for (size_t j = 0; j < row.size(); ++j) {
            out << row[j];
            if (j != row.size()-1) out << ",";
        }
        out << "\n";
    }
    out.close();
}

double calculate_normalized_entropy(const Grid& data) {
    const int bins = 20;
    const double bin_size = 1.0 / bins;
    const double log_bins = log(bins);
    int hist[bins] = {0};  // Array estilo C estático
    const double total = N*N;
    const double inv_total = 1.0 / total;

    // Usamos atomic para evitar condiciones de carrera
    #pragma omp parallel for
    for (const auto& row : data) {
        for (double val : row) {
            int bin = min(bins-1, static_cast<int>(val / bin_size));
            #pragma omp atomic
            hist[bin]++;
        }
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

double calculate_average_gradient(const Grid& data) {
    double total_gradient = 0.0;
    int gradient_count = 0;
    const int N_minus_1 = N - 1;

    #pragma omp parallel for reduction(+:total_gradient, gradient_count)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N_minus_1; ++j) {
            total_gradient += abs(data[i][j+1] - data[i][j]);
            gradient_count++;
        }
    }

    #pragma omp parallel for reduction(+:total_gradient, gradient_count)
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N_minus_1; ++i) {
            total_gradient += abs(data[i+1][j] - data[i][j]);
            gradient_count++;
        }
    }

    return gradient_count ? total_gradient / gradient_count : 0.0;
}

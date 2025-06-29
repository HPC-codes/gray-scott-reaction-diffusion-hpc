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
#include <mpi.h>

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
void initialize_BZ(Grid& u, Grid& v, int geometry_type, int num_sources, int rank, int size);
inline double laplacian(const Grid& g, int i, int j, int local_N);
void create_directory(const string& path);
void save_grid(const Grid& v, int iter, const string& output_dir, int rank, int size);
double calculate_normalized_entropy(const Grid& data, int local_N);
double calculate_average_gradient(const Grid& data, int local_N);
void print_geometry_options();
void exchange_ghost_rows(Grid& grid, int rank, int size, int local_N, MPI_Comm comm);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Variables para medición de tiempos
    double init_time = 0.0;
    double simulation_time = 0.0;
    double save_time = 0.0;
    double entropy_time = 0.0;
    double gradient_time = 0.0;
    double metrics_time = 0.0;
    double total_time = 0.0;
    double comm_time = 0.0;

    auto total_start = high_resolution_clock::now();

    if (rank == 0) {
        cout << "Tamaño de la malla (N x N, recomendado 100-2000): ";
        cin >> N;

        if (N <= 0) {
            cerr << "Error: El tamaño de la malla debe ser positivo." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        print_geometry_options();

        int geometry_type, num_sources;
        cout << "Seleccione el tipo de geometría (1-5): ";
        cin >> geometry_type;

        if (geometry_type < 1 || geometry_type > 5) {
            cerr << "Error: Opción de geometría no válida." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        if (geometry_type == 1) {
            cout << "Número de focos a crear: ";
            cin >> num_sources;
        } else {
            num_sources = 1;
        }

        // Broadcast parameters to all processes
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&geometry_type, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&num_sources, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&geometry_type, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&num_sources, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    dx_sq = dx * dx;

    // Calculate local grid dimensions
    int rows_per_proc = N / size;
    int remainder = N % size;
    int local_N = rows_per_proc + (rank < remainder ? 1 : 0);
    int start_row = rank * rows_per_proc + min(rank, remainder);
    
    // Add ghost rows (2 extra rows - one above and one below)
    int local_N_with_ghost = local_N + 2;

    string output_dir = "BZ_Geometry_" + to_string(geometry_type);
    if (rank == 0) {
        create_directory(output_dir);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Inicialización
    Grid u(local_N_with_ghost, vector<double>(N, 1.0));
    Grid v(local_N_with_ghost, vector<double>(N, 0.0));

    auto init_start = high_resolution_clock::now();
    initialize_BZ(u, v, geometry_type, num_sources, rank, size);
    auto init_end = high_resolution_clock::now();
    init_time = duration_cast<duration<double>>(init_end - init_start).count();

    if (rank == 0) {
        cout << "\n=== Simulación Belousov-Zhabotinsky con Geometrías Personalizadas ===\n";
        cout << "Tamaño: " << N << "x" << N << " | Pasos: " << steps << " | Procesos: " << size << "\n";
        cout << "Geometría seleccionada: ";
        switch(geometry_type) {
            case 1: cout << num_sources << " focos circulares"; break;
            case 2: cout << "Línea horizontal central"; break;
            case 3: cout << "Cuadrado central"; break;
            case 4: cout << "Patrón hexagonal"; break;
            case 5: cout << "Cruz central"; break;
        }
        cout << "\nSalida: " << output_dir << "\n\n";
    }

    // Configuración de métricas
    auto metrics_start = high_resolution_clock::now();
    ofstream metrics;
    if (rank == 0) {
        metrics.open(output_dir + "/metrics.csv");
        metrics << "Paso,Entropia,GradientePromedio\n";
    }

    // Métricas iniciales
    double initial_entropy = 0.0;
    double initial_grad = 0.0;
    
    if (rank == 0) {
        auto entropy_start = high_resolution_clock::now();
        initial_entropy = calculate_normalized_entropy(v, local_N);
        auto entropy_end = high_resolution_clock::now();
        entropy_time += duration_cast<duration<double>>(entropy_end - entropy_start).count();

        auto gradient_start = high_resolution_clock::now();
        initial_grad = calculate_average_gradient(v, local_N);
        auto gradient_end = high_resolution_clock::now();
        gradient_time += duration_cast<duration<double>>(gradient_end - gradient_start).count();

        metrics << 0 << "," << fixed << setprecision(6) << initial_entropy << "," << initial_grad << "\n";
        cout << "Entropía inicial: " << initial_entropy << "\n";
    }
    auto metrics_end = high_resolution_clock::now();
    metrics_time = duration_cast<duration<double>>(metrics_end - metrics_start).count();

    // Simulación principal
    auto sim_start = high_resolution_clock::now();
    for (int n = 1; n <= steps; ++n) {
        // Exchange ghost rows before computation
        auto comm_start = high_resolution_clock::now();
        exchange_ghost_rows(u, rank, size, local_N, MPI_COMM_WORLD);
        exchange_ghost_rows(v, rank, size, local_N, MPI_COMM_WORLD);
        auto comm_end = high_resolution_clock::now();
        comm_time += duration_cast<duration<double>>(comm_end - comm_start).count();

        Grid u_next = u;
        Grid v_next = v;

        // Note: We start from 1 and end at local_N to skip ghost rows
        for (int i = 1; i <= local_N; ++i) {
            for (int j = 0; j < N; ++j) {
                double u_val = u[i][j];
                double v_val = v[i][j];
                double uvv = u_val * v_val * v_val;

                double lap_u = laplacian(u, i, j, local_N);
                double lap_v = laplacian(v, i, j, local_N);

                u_next[i][j] = max(0.0, min(1.5, u_val + dt * (Du * lap_u - uvv + F * (1.0 - u_val))));
                v_next[i][j] = max(0.0, min(1.0, v_val + dt * (Dv * lap_v + uvv - F_plus_k * v_val)));
            }
        }
        u = move(u_next);
        v = move(v_next);

        if (n % output_interval == 0) {
            auto save_start = high_resolution_clock::now();
            save_grid(v, n, output_dir, rank, size);
            auto save_end = high_resolution_clock::now();
            save_time += duration_cast<duration<double>>(save_end - save_start).count();

            // Calculate metrics (only rank 0 needs the full grid)
            if (rank == 0) {
                auto metrics_step_start = high_resolution_clock::now();
                
                auto entropy_step_start = high_resolution_clock::now();
                double entropy = calculate_normalized_entropy(v, local_N);
                auto entropy_step_end = high_resolution_clock::now();
                entropy_time += duration_cast<duration<double>>(entropy_step_end - entropy_step_start).count();

                auto gradient_step_start = high_resolution_clock::now();
                double avg_grad = calculate_average_gradient(v, local_N);
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
    }
    auto sim_end = high_resolution_clock::now();
    simulation_time = duration_cast<duration<double>>(sim_end - sim_start).count();

    if (rank == 0) {
        metrics.close();
    }

    auto total_end = high_resolution_clock::now();
    total_time = duration_cast<duration<double>>(total_end - total_start).count();

    if (rank == 0) {
        cout << "\n\n=== Resultados ===\n";
        cout << "=== Tiempos de ejecución ===\n";
        cout << "Inicialización: " << fixed << setprecision(4) << init_time << " s\n";
        cout << "Simulación principal: " << simulation_time << " s\n";
        cout << "Comunicación: " << comm_time << " s\n";
        cout << "Guardado de datos: " << save_time << " s\n";
        cout << "Cálculo de entropía: " << entropy_time << " s\n";
        cout << "Cálculo de gradiente: " << gradient_time << " s\n";
        cout << "Métricas y escritura: " << metrics_time << " s\n";
        cout << "---------------------------------\n";
        cout << "Suma de tiempos parciales: "
             << (init_time + simulation_time + comm_time + save_time + entropy_time + gradient_time + metrics_time)
             << " s\n";
        cout << "Tiempo total medido: " << total_time << " s\n";
        cout << "Datos guardados en:\n";
        cout << "- " << output_dir << "/bz_XXXXX.csv (patrones espaciales)\n";
        cout << "- " << output_dir << "/metrics.csv (métricas cuantitativas)\n";
    }

    MPI_Finalize();
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

void initialize_BZ(Grid& u, Grid& v, int geometry_type, int num_sources, int rank, int size) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    // Calculate local grid dimensions
    int rows_per_proc = N / size;
    int remainder = N % size;
    int local_N = rows_per_proc + (rank < remainder ? 1 : 0);
    int start_row = rank * rows_per_proc + min(rank, remainder);
    int end_row = start_row + local_N - 1;

    // Inicialización base (including ghost rows)
    for (int i = 0; i < u.size(); ++i) {
        for (int j = 0; j < N; ++j) {
            u[i][j] = 0.8 + 0.05 * dis(gen);
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

                // Check if this process has any of these rows
                if (max_i < start_row || min_i > end_row) continue;

                for (int i = max(start_row, min_i); i <= min(end_row, max_i); ++i) {
                    double dx = i - cx;
                    // Adjust for ghost rows (local index is i - start_row + 1)
                    int local_i = i - start_row + 1;
                    for (int j = min_j; j <= max_j; ++j) {
                        double dy = j - cy;
                        if (dx*dx + dy*dy < radius_sq) {
                            v[local_i][j] = 0.9;
                            u[local_i][j] = 0.2;
                        }
                    }
                }
            }
            break;
        }
        case 2: { // Línea horizontal
            int j_start = max(0, static_cast<int>(center-3));
            int j_end = min(N-1, static_cast<int>(center+3));

            // Check if center row is in this process's range
            int center_row = static_cast<int>(center);
            if (center_row >= start_row && center_row <= end_row) {
                int local_i = center_row - start_row + 1;
                for (int j = j_start; j <= j_end; ++j) {
                    v[local_i][j] = 0.9;
                    u[local_i][j] = 0.2;
                }
            }
            break;
        }
        case 3: { // Cuadrado central
            int size_sq = N/4;
            int i_start = max(0, static_cast<int>(center-size_sq));
            int i_end = min(N-1, static_cast<int>(center+size_sq));
            int j_start = i_start, j_end = i_end;

            // Check if this process has any of these rows
            if (i_end < start_row || i_start > end_row) break;

            for (int i = max(start_row, i_start); i <= min(end_row, i_end); ++i) {
                int local_i = i - start_row + 1;
                for (int j = j_start; j <= j_end; ++j) {
                    v[local_i][j] = 0.9;
                    u[local_i][j] = 0.2;
                }
            }
            break;
        }
        case 4: { // Hexágono
            int i_start = max(0, static_cast<int>(center-hex_size));
            int i_end = min(N-1, static_cast<int>(center+hex_size));
            int j_start = max(0, static_cast<int>(center-hex_const));
            int j_end = min(N-1, static_cast<int>(center+hex_const));

            // Check if this process has any of these rows
            if (i_end < start_row || i_start > end_row) break;

            for (int i = max(start_row, i_start); i <= min(end_row, i_end); ++i) {
                int local_i = i - start_row + 1;
                double dx_val = abs(i - center);
                for (int j = j_start; j <= j_end; ++j) {
                    double dy_val = abs(j - center);
                    if (dx_val <= hex_size && dy_val <= hex_const &&
                        (0.5*hex_size + 0.866*dy_val) <= hex_size) {
                        v[local_i][j] = 0.9;
                        u[local_i][j] = 0.2;
                    }
                }
            }
            break;
        }
        case 5: { // Cruz
            int center_start = max(0, static_cast<int>(center-2));
            int center_end = min(N-1, static_cast<int>(center+2));

            // Parte horizontal - all processes participate
            for (int i = max(start_row, 0); i <= min(end_row, N-1); ++i) {
                int local_i = i - start_row + 1;
                for (int j = center_start; j <= center_end; ++j) {
                    v[local_i][j] = 0.9;
                    u[local_i][j] = 0.2;
                }
            }

            // Parte vertical - only processes that have the center columns
            for (int j = center_start; j <= center_end; ++j) {
                for (int i = max(start_row, 0); i <= min(end_row, N-1); ++i) {
                    int local_i = i - start_row + 1;
                    v[local_i][j] = 0.9;
                    u[local_i][j] = 0.2;
                }
            }
            break;
        }
    }

    // Pequeña perturbación en el resto de la matriz v (including ghost rows)
    for (auto& row : v) {
        for (auto& val : row) {
            if (val == 0.0) val = 0.001 * dis(gen);
        }
    }
}

inline double laplacian(const Grid& g, int i, int j, int local_N) {
    // Note: i is the local index including ghost rows (1..local_N)
    return (g[i+1][j] + g[i-1][j] +
            g[i][(j+1)%N] + g[i][(j-1+N)%N] -
            4.0 * g[i][j]) / dx_sq;
}

void exchange_ghost_rows(Grid& grid, int rank, int size, int local_N, MPI_Comm comm) {
    MPI_Status status;
    int up = rank - 1;
    int down = rank + 1;
    
    // Handle boundaries
    if (up < 0) up = MPI_PROC_NULL;
    if (down >= size) down = MPI_PROC_NULL;
    
    // Send to up, receive from down
    if (down != MPI_PROC_NULL) {
        MPI_Send(&grid[local_N][0], N, MPI_DOUBLE, down, 0, comm);
    }
    if (up != MPI_PROC_NULL) {
        MPI_Recv(&grid[0][0], N, MPI_DOUBLE, up, 0, comm, &status);
    }
    
    // Send to down, receive from up
    if (up != MPI_PROC_NULL) {
        MPI_Send(&grid[1][0], N, MPI_DOUBLE, up, 1, comm);
    }
    if (down != MPI_PROC_NULL) {
        MPI_Recv(&grid[local_N+1][0], N, MPI_DOUBLE, down, 1, comm, &status);
    }
}

void save_grid(const Grid& v, int iter, const string& output_dir, int rank, int size) {
    // Only rank 0 collects all data and saves to file
    if (rank == 0) {
        // Create a full grid
        Grid full_grid(N, vector<double>(N));
        
        // Copy rank 0's portion (excluding ghost rows)
        for (int i = 1; i < v.size()-1; ++i) {
            full_grid[i-1] = v[i];
        }
        
        // Receive data from other ranks
        for (int src = 1; src < size; ++src) {
            // Calculate how many rows this source has
            int rows_per_proc = N / size;
            int remainder = N % size;
            int src_rows = rows_per_proc + (src < remainder ? 1 : 0);
            int src_start = src * rows_per_proc + min(src, remainder);
            
            // Receive each row individually
            for (int i = 0; i < src_rows; ++i) {
                MPI_Recv(&full_grid[src_start + i][0], N, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        
        // Now save to file
        string filename = output_dir + "/bz_" + to_string(iter) + ".csv";
        ofstream out(filename);

        if (!out.is_open()) {
            cerr << "Error al crear archivo: " << filename << endl;
            return;
        }

        out << fixed << setprecision(6);
        for (const auto& row : full_grid) {
            for (size_t j = 0; j < row.size(); ++j) {
                out << row[j];
                if (j != row.size()-1) out << ",";
            }
            out << "\n";
        }
        out.close();
    } else {
        // Send our portion to rank 0 (excluding ghost rows)
        for (int i = 1; i < v.size()-1; ++i) {
            MPI_Send(&v[i][0], N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }
}

double calculate_normalized_entropy(const Grid& data, int local_N) {
    const int bins = 20;
    const double bin_size = 1.0 / bins;
    const double log_bins = log(bins);
    vector<int> hist(bins, 0);
    const double total = N*N;
    const double inv_total = 1.0 / total;

    // Only count non-ghost rows (from 1 to local_N)
    for (int i = 1; i <= local_N; ++i) {
        for (double val : data[i]) {
            int bin = min(bins-1, static_cast<int>(val / bin_size));
            hist[bin]++;
        }
    }

    // Sum histograms from all processes
    vector<int> global_hist(bins, 0);
    MPI_Reduce(hist.data(), global_hist.data(), bins, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank != 0) return 0.0;

    double entropy = 0.0;
    for (int count : global_hist) {
        if (count > 0) {
            double p = count * inv_total;
            entropy -= p * log(p);
        }
    }
    return entropy / log_bins;
}

double calculate_average_gradient(const Grid& data, int local_N) {
    double total_gradient = 0.0;
    int gradient_count = 0;
    const int N_minus_1 = N - 1;

    // Calculate gradients within our portion (excluding ghost rows)
    for (int i = 1; i <= local_N; ++i) {
        for (int j = 0; j < N_minus_1; ++j) {
            total_gradient += abs(data[i][j+1] - data[i][j]);
            gradient_count++;
        }
    }

    for (int j = 0; j < N; ++j) {
        for (int i = 1; i < local_N; ++i) {
            total_gradient += abs(data[i+1][j] - data[i][j]);
            gradient_count++;
        }
    }

    // Sum across all processes
    double global_total = 0.0;
    int global_count = 0;
    MPI_Reduce(&total_gradient, &global_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&gradient_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    return global_count ? global_total / global_count : 0.0;
}

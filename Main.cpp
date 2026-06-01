#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <future>
#include <filesystem>
#include <stdexcept>
#include <mutex>

namespace fs = std::filesystem;

// ==========================================
// 1. DATA STRUCTURES
// ==========================================
struct Instance {
    int n_jobs;
    int n_machines;
    std::vector<std::vector<int>> processing_times;
    std::vector<std::vector<int>> machine_sequence;
};

struct Result {
    std::string instance_name;
    int cmax = 0;
    double execution_time = 0.0;
    std::string error;
};

// ==========================================
// 2. TAILLARD PARSER
// ==========================================
Instance read_taillard_instance(const fs::path& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("File not found: " + filepath.string());
    }

    Instance inst;
    std::string line;
    
    // Read header
    while (std::getline(file, line)) {
        if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        std::stringstream ss(line);
        if (ss >> inst.n_jobs >> inst.n_machines) break;
    }

    if (inst.n_jobs <= 0 || inst.n_machines <= 0) {
        throw std::runtime_error("Number of jobs and machines must be greater than zero.");
    }

    inst.processing_times.resize(inst.n_jobs, std::vector<int>(inst.n_machines));
    inst.machine_sequence.resize(inst.n_jobs, std::vector<int>(inst.n_machines));

    // Read jobs
    for (int i = 0; i < inst.n_jobs; ++i) {
        while (std::getline(file, line)) {
            if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
            break;
        }

        std::stringstream ss(line);
        std::vector<int> tokens;
        int val;
        while (ss >> val) tokens.push_back(val);

        int expected_cols = 2 * inst.n_machines;
        if (tokens.size() != static_cast<size_t>(expected_cols)) {
            throw std::runtime_error("Invalid job line: expected " + std::to_string(expected_cols) + " values.");
        }

        int op_idx = 0;
        for (size_t j = 0; j < tokens.size(); j += 2) {
            inst.machine_sequence[i][op_idx] = tokens[j];
            inst.processing_times[i][op_idx] = tokens[j + 1];
            op_idx++;
        }
    }

    return inst;
}

// ==========================================
// 3. VND ENGINE & HEURISTICS
// ==========================================
int evaluate_sequence(const std::vector<int>& sequence, const Instance& inst) {
    std::vector<int> job_next_op(inst.n_jobs, 0);
    std::vector<int> machine_free_time(inst.n_machines, 0);
    std::vector<int> job_ready_time(inst.n_jobs, 0);

    for (int job_id : sequence) {
        int op_idx = job_next_op[job_id];
        if (op_idx >= inst.n_machines) continue;

        int machine_id = inst.machine_sequence[job_id][op_idx];
        int duration = inst.processing_times[job_id][op_idx];

        int start_time = std::max(machine_free_time[machine_id], job_ready_time[job_id]);
        int end_time = start_time + duration;

        machine_free_time[machine_id] = end_time;
        job_ready_time[job_id] = end_time;
        job_next_op[job_id]++;
    }

    return *std::max_element(machine_free_time.begin(), machine_free_time.end());
}

std::vector<int> generate_initial_solution(const Instance& inst) {
    std::vector<std::pair<int, int>> job_work;
    for (int j = 0; j < inst.n_jobs; ++j) {
        int sum = std::accumulate(inst.processing_times[j].begin(), inst.processing_times[j].end(), 0);
        job_work.push_back({j, sum});
    }

    // MWKR logic: Sort descending by total workload
    std::stable_sort(job_work.begin(), job_work.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    std::vector<int> initial_seq;
    initial_seq.reserve(inst.n_jobs * inst.n_machines);
    for (int m = 0; m < inst.n_machines; ++m) {
        for (const auto& j : job_work) {
            initial_seq.push_back(j.first);
        }
    }
    return initial_seq;
}

std::pair<std::vector<int>, int> variable_neighborhood_descent_raw(std::vector<int> current_seq, const Instance& inst) {
    int current_cmax = evaluate_sequence(current_seq, inst);
    int k = 1;
    const int max_k = 3;
    const size_t seq_len = current_seq.size();

    while (k <= max_k) {
        bool improved = false;

        if (k == 1) {
            // N1: Adjacent Swap — Best Improvement
            // Scans ALL adjacent pairs and picks the single best move.
            size_t best_i = seq_len; // sentinel: no improvement yet
            int    best_cmax = current_cmax;

            for (size_t i = 0; i < seq_len - 1; ++i) {
                if (current_seq[i] != current_seq[i + 1]) {
                    std::swap(current_seq[i], current_seq[i + 1]);
                    int cmax = evaluate_sequence(current_seq, inst);
                    std::swap(current_seq[i], current_seq[i + 1]); // always revert

                    if (cmax < best_cmax) {
                        best_cmax = cmax;
                        best_i    = i;
                    }
                }
            }

            if (best_i < seq_len) { // a strictly improving move was found
                std::swap(current_seq[best_i], current_seq[best_i + 1]);
                current_cmax = best_cmax;
                improved = true;
            }
        }
        else if (k == 2) {
            // N2: General Swap — Best Improvement
            // Scans ALL (i,j) pairs and picks the single best swap.
            size_t best_i = seq_len, best_j = seq_len; // sentinels
            int    best_cmax = current_cmax;

            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = i + 1; j < seq_len; ++j) {
                    if (current_seq[i] != current_seq[j]) {
                        std::swap(current_seq[i], current_seq[j]);
                        int cmax = evaluate_sequence(current_seq, inst);
                        std::swap(current_seq[i], current_seq[j]); // always revert

                        if (cmax < best_cmax) {
                            best_cmax = cmax;
                            best_i    = i;
                            best_j    = j;
                        }
                    }
                }
            }

            if (best_i < seq_len) {
                std::swap(current_seq[best_i], current_seq[best_j]);
                current_cmax = best_cmax;
                improved = true;
            }
        }
        else if (k == 3) {
            // N3: Insertion/Shift — Best Improvement
            // Scans ALL (i,j) removal+insertion pairs and picks the single best move.
            size_t best_i = seq_len, best_j = seq_len; // sentinels
            int    best_cmax = current_cmax;

            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j < seq_len; ++j) {
                    if (i != j) {
                        std::vector<int> neighbor = current_seq;
                        int val = neighbor[i];
                        neighbor.erase(neighbor.begin() + i);
                        neighbor.insert(neighbor.begin() + j, val);

                        int cmax = evaluate_sequence(neighbor, inst);
                        if (cmax < best_cmax) {
                            best_cmax = cmax;
                            best_i    = i;
                            best_j    = j;
                        }
                    }
                }
            }

            if (best_i < seq_len) {
                int val = current_seq[best_i];
                current_seq.erase(current_seq.begin() + best_i);
                current_seq.insert(current_seq.begin() + best_j, val);
                current_cmax = best_cmax;
                improved = true;
            }
        }

        if (improved) {
            k = 1; // Improvement found: restart from the lightest neighborhood
        } else {
            k += 1; // No improvement: move to the next, more complex neighborhood
        }
    }

    return {current_seq, current_cmax};
}

Result solve_instance_safe(const fs::path& path) {
    Result res;
    res.instance_name = path.filename().string();
    
    auto start_time = std::chrono::high_resolution_clock::now();
    try {
        Instance inst = read_taillard_instance(path);
        std::vector<int> initial_seq = generate_initial_solution(inst);
        auto [best_seq, best_cmax] = variable_neighborhood_descent_raw(initial_seq, inst);
        
        res.cmax = best_cmax;
    } catch (const std::exception& error) {
        res.error = error.what();
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> duration = end_time - start_time;
    res.execution_time = duration.count();
    
    return res;
}

// ==========================================
// 4. PARALLEL RUNNER & CLI
// ==========================================
std::mutex print_mutex;

void run_single_instance(const fs::path& instance_path) {
    Result res = solve_instance_safe(instance_path);
    if (res.error.empty()) {
        std::cout << "Instance: " << instance_path.string() << "\n"
                  << "Makespan (C_max): " << res.cmax << "\n"
                  << "Time: " << std::fixed << std::setprecision(3) << res.execution_time << "s\n";
    } else {
        std::cerr << "Error processing " << instance_path.string() << ": " << res.error << "\n";
    }
}

void run_group_and_save(int start_idx, int end_idx, const std::string& output_file) {
    std::vector<fs::path> instance_paths;
    for (int idx = start_idx; idx <= end_idx; ++idx) {
        std::stringstream ss;
        ss << "Data/ta" << std::setfill('0') << std::setw(2) << idx;
        instance_paths.push_back(fs::path(ss.str()));
    }

    std::cout << "Starting parallel processing on up to " 
              << std::thread::hardware_concurrency() << " cores...\n";

    std::vector<std::future<Result>> futures;
    for (const auto& path : instance_paths) {
        // Launch tasks asynchronously
        futures.push_back(std::async(std::launch::async, solve_instance_safe, path));
    }

    std::vector<Result> results;
    long long total_cmax = 0;
    int success_count = 0;

    for (auto& future : futures) {
        Result res = future.get(); // Wait for completion
        results.push_back(res);

        std::lock_guard<std::mutex> lock(print_mutex);
        if (res.error.empty()) {
            std::cout << res.instance_name << " (Completed): C_max = " << res.cmax 
                      << " | Time: " << std::fixed << std::setprecision(3) << res.execution_time << "s\n";
            total_cmax += res.cmax;
            success_count++;
        } else {
            std::cout << res.instance_name << " (Error): " << res.error << "\n";
        }
    }

    // Save results
    std::ofstream out(output_file);
    if (!out.is_open()) {
        std::cerr << "Failed to open output file: " << output_file << "\n";
        return;
    }

    for (const auto& res : results) {
        if (res.error.empty()) {
            out << res.instance_name << ";" << res.cmax << ";" << std::fixed << std::setprecision(3) << res.execution_time << "s\n";
        } else {
            out << res.instance_name << ";ERROR;" << res.error << "\n";
        }
    }

    double avg_cmax = success_count > 0 ? static_cast<double>(total_cmax) / success_count : 0.0;
    out << "AVERAGE;" << std::fixed << std::setprecision(2) << avg_cmax << "\n";
    
    std::cout << "Results recorded in: " << output_file << "\n"
              << "Group average Makespan: " << std::fixed << std::setprecision(2) << avg_cmax << "\n";
}

fs::path parse_instance_input(const std::string& user_text) {
    if (user_text.find_first_not_of("0123456789") == std::string::npos) {
        std::stringstream ss;
        ss << "Data/ta" << std::setfill('0') << std::setw(2) << std::stoi(user_text);
        return fs::path(ss.str());
    }
    return fs::path(user_text);
}

void run_menu() {
    while (true) {
        std::cout << "\n===== JOB SHOP MENU (MWKR -> VND RAW) =====\n"
                  << "1 - Run single instance\n"
                  << "2 - Run group 41 to 50\n"
                  << "3 - Run group 71 to 80\n"
                  << "4 - Exit\n"
                  << "Choose an option: ";

        std::string option;
        std::getline(std::cin, option);

        if (option == "1") {
            std::cout << "Enter the instance number (e.g., 41) or path (e.g., Data/ta41): ";
            std::string user_input;
            std::getline(std::cin, user_input);
            try {
                run_single_instance(parse_instance_input(user_input));
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << "\n";
            }
        } 
        else if (option == "2") run_group_and_save(41, 50, "results_group_41_50.txt");
        else if (option == "3") run_group_and_save(71, 80, "results_group_71_80.txt");
        else if (option == "4") {
            std::cout << "Terminating execution.\n";
            break;
        } 
        else std::cout << "Invalid option. Please try again.\n";
    }
}

int main() {
    run_menu();
    return 0;
}
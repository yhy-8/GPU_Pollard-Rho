#include <stdio.h>
#include <cuda_runtime.h>
#include <gmp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include "cgbn/cgbn.h"

// ================= 配置区域 =================
#define BITS 1024              
#define TPI 32                 
#define MAX_ITERATIONS 20000000 
#define TPB 128                

typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;
typedef typename env_t::cgbn_t bn_t;

typedef struct {
    cgbn_mem_t<BITS> n;      
    cgbn_mem_t<BITS> factor; 
    int found;               
    int gpu_id;              
} gpu_data_t;

// ================= CUDA Kernel =================
__global__ void pollard_rho_kernel(gpu_data_t *data, uint32_t count) {
    context_t bn_context(cgbn_report_monitor); 
    env_t bn_env(bn_context.env<env_t>());
    int32_t instance_id = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
    
    if (instance_id >= count) return;

    bn_t n, x, y, d, c, t, abs_diff;
    cgbn_load(bn_env, n, &(data->n)); 

    if (data->found) return;

    cgbn_set_ui32(bn_env, x, 2);
    cgbn_set_ui32(bn_env, y, 2);
    cgbn_set_ui32(bn_env, c, instance_id + 1); 
    cgbn_set_ui32(bn_env, d, 1);
    bn_t one;
    cgbn_set_ui32(bn_env, one, 1);

    for (int i = 0; i < MAX_ITERATIONS; ++i) {
        if (data->found) break; 

        cgbn_mul(bn_env, t, x, x);
        cgbn_add(bn_env, t, t, c);
        cgbn_rem(bn_env, x, t, n);

        cgbn_mul(bn_env, t, y, y);
        cgbn_add(bn_env, t, t, c);
        cgbn_rem(bn_env, y, t, n);
        cgbn_mul(bn_env, t, y, y);
        cgbn_add(bn_env, t, t, c);
        cgbn_rem(bn_env, y, t, n);

        if (cgbn_compare(bn_env, x, y) >= 0) cgbn_sub(bn_env, abs_diff, x, y);
        else cgbn_sub(bn_env, abs_diff, y, x);
        
        cgbn_gcd(bn_env, d, abs_diff, n);

        if (cgbn_compare(bn_env, d, one) > 0) {
            if (cgbn_compare(bn_env, d, n) == 0) {
                break; 
            } else {
                if (atomicCAS(&(data->found), 0, 1) == 0) {
                    cgbn_store(bn_env, &(data->factor), d);
                    data->gpu_id = instance_id;
                }
                break;
            }
        }
    }
}

void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

bool run_gpu_factorization(mpz_t n, mpz_t result, gpu_data_t* d_data) {
    gpu_data_t h_data;
    memset(&h_data, 0, sizeof(gpu_data_t)); 

    size_t countp;
    mpz_export(h_data.n._limbs, &countp, -1, sizeof(uint32_t), 0, 0, n);
    
    checkCudaErrors(cudaMemcpy(d_data, &h_data, sizeof(gpu_data_t), cudaMemcpyHostToDevice));

    uint32_t num_instances = 4096; 
    uint32_t threads_per_block = TPB;
    uint32_t blocks = (num_instances * TPI + threads_per_block - 1) / threads_per_block;

    pollard_rho_kernel<<<blocks, threads_per_block>>>(d_data, num_instances);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(&h_data, d_data, sizeof(gpu_data_t), cudaMemcpyDeviceToHost));

    if (h_data.found) {
        mpz_import(result, (BITS/32), -1, sizeof(uint32_t), 0, 0, h_data.factor._limbs);
        return true;
    }
    return false;
}

bool stringNumCompare(const std::string& a, const std::string& b) {
    if (a.length() != b.length()) return a.length() < b.length();
    return a < b;
}

// ================= 小因子试除逻辑 =================
// 返回 true 表示 n 已经被完全分解为1了
// 返回 false 表示还有剩余部分需要 GPU 处理
bool trial_division(mpz_t n, std::vector<std::string>& primes) {
    // 试除 2 到 1000 之间的小数
    // 这样可以解决 4, 6, 9 等小合数问题，也能加速大数分解
    for (unsigned long i = 2; i <= 1000; i++) {
        if (mpz_cmp_ui(n, 1) <= 0) return true; // 如果已经是1了，结束

        while (mpz_divisible_ui_p(n, i)) {
            // 能被 i 整除
            mpz_divexact_ui(n, n, i); // n = n / i
            primes.push_back(std::to_string(i));
            // printf("[CPU Trial] Found small factor: %lu\n", i);
        }
    }
    return (mpz_cmp_ui(n, 1) <= 0);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <number>\n", argv[0]);
        return 1;
    }

    printf("========================================\n");
    printf("                Config     \n");
    printf("========================================\n");
    // %-16s 表示字符串左对齐，占16位宽
    // %d    表示输出整数
    printf("%-16s: %d\n", "BITS", BITS);
    printf("%-16s: %d\n", "TPI", TPI);
    printf("%-16s: %d\n", "TPB", TPB);
    printf("%-16s: %d\n", "MAX_ITERATIONS", MAX_ITERATIONS);
    printf("========================================\n");

     // 使用 GMP 判断二进制位数 ---
    mpz_t n;
    // 初始化并将输入的十进制字符串转换为 GMP 整数
    // mpz_init_set_str 返回 0 表示转换成功，-1 表示失败
    if (mpz_init_set_str(n, argv[1], 10) == 0) {
        // base设为2，直接获取二进制的位数
        size_t inputbits = mpz_sizeinbase(n, 2);
        printf("Inputnum BitLength: %zu\n", inputbits);
        printf("========================================\n");
        
        // 记得释放 GMP 变量占用的内存
        mpz_clear(n);
    } else {
        fprintf(stderr, "Error: Invalid number format.\n");
        return 1;
    }

    gpu_data_t *d_data;
    checkCudaErrors(cudaMalloc((void **)&d_data, sizeof(gpu_data_t)));

    std::vector<std::string> composites; 
    std::vector<std::string> primes;     
    
    composites.push_back(argv[1]);

    mpz_t current_n, factor, quotient;
    mpz_init(current_n);
    mpz_init(factor);
    mpz_init(quotient);

    printf("Starting Full Factorization for: %s\n", argv[1]);
    printf("Strategy: CPU Trial Division -> Miller-Rabin -> GPU Pollard's Rho\n");
    printf("========================================\n");

    while (!composites.empty()) {
        std::string s_n = composites.back();
        composites.pop_back();
        mpz_set_str(current_n, s_n.c_str(), 10);

        // 1. 检查是否为 1
        if (mpz_cmp_ui(current_n, 1) <= 0) continue;

        // 2. CPU 试除法 (解决小合数问题)
        // 如果 n 是 4，这里会除以2两次，n变为1，函数返回true，直接continue进入下一次循环
        if (trial_division(current_n, primes)) {
            continue; 
        }

        // 经过试除后，current_n 可能变小了，再次检查素性
        // 3. 素性测试
        if (mpz_probab_prime_p(current_n, 25) > 0) {
            char* tmp = mpz_get_str(NULL, 10, current_n);
            primes.push_back(std::string(tmp));
            printf("[Found Prime]: %s\n", tmp);
            free(tmp);
            continue;
        }

        // 4. 调用 GPU 分解
        char* s_current_str = mpz_get_str(NULL, 10, current_n);
        printf("Factoring composite on GPU: %s ... ", s_current_str);
        fflush(stdout);

        if (run_gpu_factorization(current_n, factor, d_data)) {
            mpz_div(quotient, current_n, factor); 

            char* s_f = mpz_get_str(NULL, 10, factor);
            char* s_q = mpz_get_str(NULL, 10, quotient);
            
            printf("Found factor!\n");
            printf("   -> Split into: %s * %s\n", s_f, s_q);

            composites.push_back(std::string(s_f));
            composites.push_back(std::string(s_q));

            free(s_f);
            free(s_q);
        } else {
            printf("Failed (Max Iterations Reached).\n");
            // 虽然失败了，但为了不丢数据，还是存入结果，并在输出时标记
            primes.push_back(std::string(s_current_str)); 
        }
        free(s_current_str);
    }

    printf("\n================ Final Result ================\n");
    std::cout << argv[1] << " = ";
    
    std::sort(primes.begin(), primes.end(), stringNumCompare);

    for (size_t i = 0; i < primes.size(); ++i) {
        std::cout << primes[i];
        if (i < primes.size() - 1) std::cout << " * ";
    }
    std::cout << std::endl;
    printf("==============================================\n");

    cudaFree(d_data);
    mpz_clear(current_n);
    mpz_clear(factor);
    mpz_clear(quotient);

    return 0;
}

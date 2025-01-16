#include <iostream>
#include <cstring>
#include <vector>
#include <iomanip>
#include <omp.h>
#include <stdlib.h>
#include <new>
#include <thread>
#include <mutex>
#include <chrono>
#include <fstream>


#define N (1u << 27)


#define CACHE_LINE 64

//#if defined (__GNUC__)&&__GNUC__<14
//    #define hardware_destructive_interference_size 64
//#else
//    #include <thread>
//    using std::hardware_destructive_interference_size
//#endif

static unsigned g_num_threads = std::thread::hardware_concurrency();

class barrier {
    std::condition_variable cv;
    std::mutex mtx;
    bool generation = false;
    unsigned T;
    const unsigned T0;

public:
    barrier(unsigned threads) : T(threads), T0(threads) {}

    void arrive_and_wait() {
        std::unique_lock<std::mutex> l(mtx);
        if (--T == 0) {
            T = T0;
            generation = !generation;
            cv.notify_all();
        }
        else {
            bool my_barrier = generation;
            cv.wait(l, [&] { return my_barrier != generation; });
        }
    }
};

struct table_row {
    bool match; double time, speedup, efficiency;
};

void set_num_threads(unsigned T) {
    g_num_threads = T;
    omp_set_num_threads(T);
};


unsigned get_num_threads() {
    return g_num_threads;
};

#if !defined (__cplusplus) || __cplusplus < 20200000
    typedef unsigned (*sum_ptr) (const unsigned* V, size_t n);
#else 
    template <class F> //#include type_traits
    concept sum_callable = std::is_invocable_r<unsigned, F, const unsigned*, size_t>;
#endif

std::vector<table_row> run_experiment(sum_ptr sum) {

    unsigned P = get_num_threads();
    std::vector<table_row> table(P);
    size_t n = 1 << 27;
    auto V = std::make_unique<unsigned[]>(n);

    for (unsigned T = 1; T <= P; ++T) {
        for (size_t i = 0; i < n; ++i) V[i] = i + T;
        auto t1 = std::chrono::steady_clock::now();
        table[T-1].match = (sum(V.get(), n) == (0xFC000000 + T * N));
        auto t2 = std::chrono::steady_clock::now();
        table[T - 1].time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        table[T - 1].speedup = table[0].time / table[T - 1].time;
        table[T - 1].efficiency = table[T - 1].speedup / T;
    }

    return table;
}

struct partial_sum_t {
    alignas(CACHE_LINE) unsigned val;
};

unsigned sum_mutex(const unsigned* v, size_t n) {
    unsigned sum = 0;
#pragma omp parallel
    {
        unsigned t = omp_get_thread_num();
        unsigned T = omp_get_num_threads();
        unsigned s_t = n / T, b_t = n % T;

        if (t < b_t)
            b_t = ++s_t * t;
        else
            b_t += s_t * t;

        unsigned e_t = b_t + s_t;

        unsigned my_sum = 0;

        for (unsigned i = b_t; i < e_t; i++)
            my_sum += v[i];
#pragma omp critical
        {
            sum += my_sum;
        }
    }
    return sum;
}

unsigned vector_sum_la(const unsigned* v, size_t n) {
    unsigned T;
    unsigned sum = 0;
    partial_sum_t* partial_sums;
#pragma omp parallel
    {
        unsigned t = omp_get_thread_num();
#pragma omp single
        {
            T = omp_get_num_threads();
            partial_sums = (partial_sum_t*)malloc(sizeof partial_sums[0] * T);
        }

        partial_sums[t].val = 0;
        unsigned s_t = n / T, b_t = n % T;

        if (t < b_t)
            b_t = ++s_t * t;
        else
            b_t += s_t * t;

        unsigned e_t = b_t + s_t;

        for (unsigned i = b_t; i < e_t; i++)
            partial_sums[t].val += v[i];
    }

    for (unsigned i = 0; i < T; i++)
        sum += partial_sums[i].val;
    free(partial_sums);
    return sum;
}

unsigned sum_round_robin_aligned(const unsigned* v, size_t n) {
    unsigned sum = 0;
    partial_sum_t* partial_sums;
    unsigned T;
#pragma omp parallel
    {
        unsigned t = omp_get_thread_num();
#pragma omp single
        {
            T = omp_get_num_threads();
            partial_sums = (partial_sum_t*)malloc(sizeof partial_sums[0] * T);
        }
        partial_sums[t].val = 0;
        for (unsigned i = t; i < n; i += T)
            partial_sums[t].val += v[i];
    }

    for (unsigned i = 0; i < T; i++)
        sum += partial_sums[i].val;
    free(partial_sums);
    return sum;
}

unsigned sum_round_robin(const unsigned* v, size_t n) {
    unsigned sum = 0;
    unsigned* partial_sums;
    unsigned T;
    #pragma omp parallel
    {
        T = omp_get_num_threads();
        unsigned t = omp_get_thread_num();
        #pragma omp single
        {
            partial_sums = (unsigned*) calloc(sizeof v[0], T);
        }

        for (unsigned i = t; i < n; i += T)
            partial_sums[t] += v[i];
    }

    for (unsigned i = 0; i < T; i++)
        sum += partial_sums[i];
    free(partial_sums);
    return sum;
}

unsigned sum_omp_reduce(const unsigned* v, size_t n) {
    unsigned sum = 0;
#pragma omp parallel for reduction(+ :sum)
    for (int i = 0; i < n; i++)
        sum += v[i];
    return sum;
}

unsigned sum_seq(const unsigned* v, size_t n) {
    unsigned sum = 0;
    for (int i = 0; i < n; i++)
        sum += v[i];
    return sum;
}

unsigned sum_cpp_cs(const unsigned* v, size_t n) //Вынести мткс из цикла
{
    unsigned sum = 0;
    unsigned T = get_num_threads(); 
    std::vector<std::thread> workers(T - 1);
    std::mutex mtx;
    auto worker_proc = [T, v, n, &sum, &mtx](unsigned t) {
        unsigned s_t = n / T, b_t = n % T;

        if (t < b_t)
            b_t = ++s_t * t;
        else
            b_t += s_t * t;

        unsigned e_t = b_t + s_t;

        unsigned my_sum = 0;

        for (unsigned i = b_t; i < e_t; i++)
            my_sum += v[i];

        {
            std::scoped_lock lock(mtx);
            sum += my_sum;
        }
    };

    for(size_t t = 1; t < T; ++t)
        workers[t - 1] = std::thread(worker_proc, t);
    worker_proc(0);

    for (auto& worker : workers)
        worker.join();

    return sum;
}

unsigned sum_seq_t(const unsigned* v, size_t n) {
    unsigned sum = 0;
    for (size_t i = 0; i < n; ++i)
        sum += v[i];
    return sum;
}

unsigned sum_barrier(const unsigned* v, size_t n) {
    unsigned sum = 0;
    unsigned T = get_num_threads();
    barrier sync_barrier(T);

    std::vector<std::thread> workers(T - 1);

    auto worker_proc = [&v, &sum, &sync_barrier, n, T](unsigned t) {
        unsigned s_t = n / T, b_t = n % T;

        if (t < b_t)
            b_t = ++s_t * t;
        else
            b_t += s_t * t;

        unsigned e_t = b_t + s_t;

        unsigned my_sum = 0;
        for (unsigned i = b_t; i < e_t; ++i)
            my_sum += v[i];

        sync_barrier.arrive_and_wait();

        {
            static std::mutex sum_mutex;
            std::scoped_lock lock(sum_mutex);
            sum += my_sum;
        }
    };

    for (unsigned t = 1; t < T; ++t)
        workers[t - 1] = std::thread(worker_proc, t);

    worker_proc(0);

    for (auto& worker : workers)
        worker.join();

    return sum;
}


int main(int argc, char** argv)  {
    set_num_threads(8);

    std::vector<table_row> tbr1 = run_experiment(sum_seq); 
    auto out1 = std::ofstream("1.csv", std::ios_base::out);
    out1 << "Match,Time,Speedup,Efficiency\n";
    for (auto t : tbr1) {
        out1 << t.match << ',' << t.time << ',' << t.speedup << ',' << t.efficiency << "\n";
    }

    std::vector<table_row> tbr2 = run_experiment(sum_seq_t);
    auto out2 = std::ofstream("2.csv", std::ios_base::out);
    out2 << "Match,Time,Speedup,Efficiency\n";
    for (auto t : tbr2) {
        out2 << t.match << ',' << t.time << ',' << t.speedup << ',' << t.efficiency << "\n";
    }

    std::vector<table_row> tbr3 = run_experiment(sum_mutex);
    auto out3 = std::ofstream("3.csv", std::ios_base::out);
    out3 << "Match,Time,Speedup,Efficiency\n";
    for (auto t : tbr3) {
        out3 << t.match << ',' << t.time << ',' << t.speedup << ',' << t.efficiency << "\n";
    }

    std::vector<table_row> tbr4 = run_experiment(vector_sum_la);
    auto out4 = std::ofstream("4.csv", std::ios_base::out);
    out4 << "Match,Time,Speedup,Efficiency\n";
    for (auto t : tbr4) {
        out4 << t.match << ',' << t.time << ',' << t.speedup << ',' << t.efficiency << "\n";
    }

    std::vector<table_row> tbr5 = run_experiment(sum_round_robin);
    auto out5 = std::ofstream("5.csv", std::ios_base::out);
    out5 << "Match,Time,Speedup,Efficiency\n";
    for (auto t : tbr5) {
        out5 << t.match << ',' << t.time << ',' << t.speedup << ',' << t.efficiency << "\n";
    }

    std::vector<table_row> tbr6 = run_experiment(sum_round_robin_aligned);
    auto out6 = std::ofstream("6.csv", std::ios_base::out);
    out6 << "Match,Time,Speedup,Efficiency\n";
    for (auto t : tbr6) {
        out6 << t.match << ',' << t.time << ',' << t.speedup << ',' << t.efficiency << "\n";
    }

    std::vector<table_row> tbr7 = run_experiment(sum_cpp_cs);
    auto out7 = std::ofstream("7.csv", std::ios_base::out);
    out7 << "Match,Time,Speedup,Efficiency\n";
    for (auto t : tbr7) {
        out7 << t.match << ',' << t.time << ',' << t.speedup << ',' << t.efficiency << "\n";
    }

    std::vector<table_row> tbr8 = run_experiment(sum_barrier);
    auto out8 = std::ofstream("8.csv", std::ios_base::out);
    out8 << "Match,Time,Speedup,Efficiency\n";
    for (auto t : tbr8) {
        out8 << t.match << ',' << t.time << ',' << t.speedup << ',' << t.efficiency << "\n";
    }

    std::vector<table_row> tbr9 = run_experiment(sum_omp_reduce);
    auto out9 = std::ofstream("9.csv", std::ios_base::out);
    out9 << "Match,Time,Speedup,Efficiency\n";
    for (auto t : tbr9) {
        out9 << t.match << ',' << t.time << ',' << t.speedup << ',' << t.efficiency << "\n";
    }

    return 0;
}
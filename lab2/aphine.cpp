#include <iostream>
#include <concepts>
#include <type_traits>
#include <memory>
#include <vector>
#include <thread>
#include <chrono>
#include <mutex>
#include <iomanip> // Для std::setw
#include <fstream>



static unsigned g_num_threads = std::thread::hardware_concurrency();

struct table_row {
    unsigned average; double time, speedup;
};

#if !defined (__cplusplus) || __cplusplus < 20200000
typedef double (*rand_ptr) (unsigned* V, size_t n, unsigned x0, unsigned x_min, unsigned x_max);
#else 
template <class F> //#include type_traits
concept sum_callable = std::is_invocable_r<unsigned, F, const unsigned*, size_t>;
#endif

void set_num_threads(unsigned T) {
    g_num_threads = T;
};

unsigned get_num_threads() {
    return g_num_threads;
};

std::vector<table_row> run_experiment(rand_ptr rand) {

    unsigned P = get_num_threads();
    std::vector<table_row> table(P);
    size_t n = 1 << 27;
    auto V = std::make_unique<unsigned[]>(n);

    for (unsigned T = 1; T <= P; ++T) {
        set_num_threads(T);
        auto t1 = std::chrono::steady_clock::now();
        table[T - 1].average = rand(V.get(), n, 3, 1000, 3000) / n;
        auto t2 = std::chrono::steady_clock::now();
        table[T - 1].time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        table[T - 1].speedup = table[0].time / table[T - 1].time;
    }

    return table;
}




template <class T>
concept monoid = requires(T x) { T(); x *= x; };

template <monoid T, std::unsigned_integral U>
T powm(T x, U e)
{
    auto r = T();

    while (e > 0)
    {
        if (e & 1)
            r *= x;
        x *= x;

        e >>= 1;
    }

    return r;
}

template <class T = unsigned>
class affine_monoid {
    T m_a = 1, m_b = 0;

public:
    affine_monoid() = default;
    affine_monoid(T a, T b): m_a(a), m_b(b) {}
    affine_monoid& operator *= (const affine_monoid& r) {
        m_b += m_a * r.m_b;
        m_a *= r.m_a;
        return *this;
    }
    T operator()(T x) const {
        return m_a * x + m_b;
    }
};

template <class T = unsigned>
class multiplicative_monoid {
    T x = T{1};

public:
    multiplicative_monoid() = default;
    explicit multiplicative_monoid(T val) : x(val) {}
    multiplicative_monoid& operator *= (const multiplicative_monoid& right) {
        x *= right.x;
        return *this;
    }

    explicit operator T() const {
        return x;
    }
};

template <class T = unsigned>
class additive_monoid {
    T x = T{0};

public:
    additive_monoid() = default;
    explicit additive_monoid(T val) : x(val) {}
    additive_monoid& operator *= (const additive_monoid& right) {
        x += right.x;
        return *this;
    }

    explicit operator T() const {
        return x;
    }
};

// void randomize (unsigned* V, size_t n, unsigned x0, unsigned x_min, unsigned x_max)
// V[i] <- (A, b)^i  *  (x[0]) = [xmin xmax]
// ранд-р x[i+1] = (A * x[i]) mod c
// рандомизатор A = 0x8088405 (134775813), b = 1, c = 2^32 (4294967296) unsigned
// линейный конгруентный генератор 
// Пар. участок:
// [beg, e) = subvector(t, T, n)
// affine_monoid m{A, b}
// m = pow(m, beg)  ((было b если че))
// for any i in [beg, e)
//   x <- my_m(x[0])
//   V <- r(x)
//   my_m *= m

unsigned range(unsigned x, unsigned xmin, unsigned xmax) {
    return xmin + (x % (xmax - xmin));
}

void seq_rand(unsigned* V, size_t n, unsigned x0, unsigned x_min, unsigned x_max) {
    const unsigned A = 0x8088405;
    const unsigned b = 1;
    const unsigned long long c = 4294967296;
    unsigned x = x0;
    V[0] = x0;
    for (size_t i = 1; i < n; i++) {
        x = (A * x + b) % c;
        V[i] = range(x,x_min, x_max);
    }
}

double randomize(unsigned* V, size_t n, unsigned x0, unsigned x_min, unsigned x_max) {
    const unsigned A = 0x8088405;
    const unsigned b = 1;
    const unsigned long long c = 4294967296;

    double sum = 0;

    unsigned T = get_num_threads();
    std::vector<std::thread> workers(T - 1);
    std::mutex mtx;

    auto subvector = [T, V, n, x0, x_min, x_max, &mtx, A, b, &sum](unsigned t) {
        unsigned x;
        unsigned s_t = n / T, b_t = n % T;
        affine_monoid m( A, b );

        if (t < b_t)
            b_t = ++s_t * t;
        else
            b_t += s_t * t;

        unsigned e_t = b_t + s_t;

        double my_sum = 0;

        affine_monoid my_m = powm(m, b_t);

        for (size_t i = b_t; i < e_t; i++)
        {
            x = my_m(x0) % c;
            V[i] = range(x, x_min, x_max);
            my_m *= m;
            my_sum += V[i];
        }

        {
            std::scoped_lock lock(mtx);
            sum += my_sum;
        }
    };

    for (size_t t = 1; t < T; ++t)
        workers[t - 1] = std::thread(subvector, t);
    subvector(0);

    for (auto& worker : workers)
        worker.join();

    return sum;
}

int main(int argc, char* argv[])
{

    std::vector<table_row> tbr1 = run_experiment(randomize);
    auto out1 = std::ofstream("randomizer.csv", std::ios_base::out);
    out1 << "Average,Time,Speedup\n";
    for (auto t : tbr1) {
        out1 << t.average << ',' << t.time << ',' << t.speedup << "\n";
    }

    /*std::cout << unsigned(powm(additive_monoid(7u), 5u)) << '\n';
    std::cout << unsigned(powm(multiplicative_monoid(7u), 5u)) << '\n';
    std::cout << unsigned(powm(multiplicative_monoid(0xffffffff), 2u)) << '\n';*/
    
    return 0;
}
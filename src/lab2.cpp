#include <bits/stdc++.h>
#include <omp.h>

using namespace std;

// input constraints
const double sigma = 0.07;
const double alpha = 3.5;
const double beta = 1.5;
const double mass = 5e-5;
const double scale = 5;
const double dt = 1e-9;

// task parameters
const int N = 500;  // size of cloud
const int M = 100; // time steps

struct vec_t {
  double x, y;
  vec_t() : x(0), y(0) {}
  vec_t(double x, double y) : x(x), y(y) {}
  vec_t operator - (const vec_t &other) const {
    return vec_t(x - other.x, y - other.y);
  }
  vec_t operator + (const vec_t &other) const {
    return vec_t(x + other.x, y + other.y);
  }
  vec_t operator * (double val) const {
    return vec_t(x * val, y * val);
  }
  vec_t operator / (double val) const {
    return vec_t(x / val, y / val);
  }
  vec_t& operator += (const vec_t &other) {
    return *this = *this + other;
  }
  vec_t& operator *= (double val) {
    return *this = *this * val;
  }
  friend ostream& operator << (ostream &os, const vec_t &v) {
    return os << v.x << " " << v.y;
  }
  double len() const {
    return hypot(x, y);
  }
  double dist_to(const vec_t &other) const {
    return (*this - other).len();
  }
};

double rnd() {
  int mx = 10000;
  return double(rand() % mx) / mx;
}

void init_random(vec_t pos[N]) {
  generate_n(pos, N, []() {
    return vec_t(rnd(), rnd());
  });
}

void init_euler(const vec_t pos[N], const vec_t fource[N], vec_t npos[N]) {
  #pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    npos[i] = pos[i] + fource[i] * (dt * dt / (2 * mass));
  }
}

void calc_fource(const vec_t pos[N], vec_t fource[N]) {
  #pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (j != i) {
        double dist = pos[i].dist_to(pos[j]);
        double v1 = alpha * pow(sigma, alpha) / pow(dist, alpha + 1);
        double v2 = beta  * pow(sigma, beta) / pow(dist, beta + 1);
        fource[i] += (pos[j] - pos[i]) * (v1 - v2);
      }
    }
    fource[i] *= scale;
  }
}

void calc_pos(const vec_t prev[N], const vec_t cur[N],
  const vec_t fource[N], vec_t npos[N]) {
  #pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    npos[i] = cur[i] * 2 - prev[i] + fource[i] * (dt * dt / mass);
  }
}

void inc(int &x) {
  ++x;
  if (x >= 3) {
    x -= 3;
  }
}

vec_t pos[3][N];
vec_t fource[3][N];

vec_t res[50][N];
int res_size = 0;

void save(int idx) {
  copy(pos[idx], pos[idx] + N, res[res_size++]);
}

void process() {
  double t0 = omp_get_wtime();
  omp_set_num_threads(4);
  init_random(pos[0]);
  calc_fource(pos[0], fource[0]);
  save(0);

  init_euler(pos[0], fource[0], pos[1]);
  calc_fource(pos[1], fource[1]);
  save(1);

  int prev = 0, cur = 1, next = 2;
  for (int t = 2; t <= M; ++t, inc(prev), inc(cur), inc(next)) {
    calc_pos(pos[prev], pos[cur], fource[cur], pos[next]);
    calc_fource(pos[next], fource[next]);
    if (t % (M / 40) == 0) {
      save(next);
    }
  }
  double t1 = omp_get_wtime();
  cerr << "time: = " << t1 - t0 << " s." << endl;
}

void out() {
  cout.setf(ios::fixed);
  cout.precision(5);
  cout << res_size << endl;
  for (int i = 0; i < res_size; ++i) {
    for (int j = 0; j < N; ++j) {
      cout << res[i][j].x << " ";
    }
    cout << endl;
    for (int j = 0; j < N; ++j) {
      cout << res[i][j].y << " ";
    }
    cout << endl;
  }
}

int main(int argc, char **argv) {
  freopen("output.txt", "w", stdout);
  process();
  out();
  return 0;
}
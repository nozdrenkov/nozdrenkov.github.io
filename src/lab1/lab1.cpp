#include <bits/stdc++.h>
#include <omp.h>

#define dbg(x) do { cerr << #x << " = " << x << endl; } while(0)

using namespace std;

// input task constraints
const double lambda = 320;
const double len = 0.18;
const double full_time = 15;
const double p = 19300;
const double c = 130;
const double q = 500;

// grid size
const int N = 1000;
const int M = 1000000;

// inner constants
const double pi = acos(-1.);
const double dx = len / N;
const double dt = full_time / M;
const double left_val = 0;
const double right_val = dx * q / lambda;
const double K = 2 * abs(lambda / (p * c)) * dt / (dx * dx);

// two last layers of the grid
double T[N + 1][2];

int res_size = 0;
double res[N + 1][20];
double res_time[20];

void save_stamp(int idx, double time) {
  for (int i = 0; i <= N; ++i) {
    res[i][res_size] = T[i][idx];
  }
  res_time[res_size] = time;
  res_size++;
}

void process() {
  omp_set_num_threads(4);
  double t0 = omp_get_wtime();

  #pragma parallel for
  for (int i = 0; i <= N; ++i) {
    double x = i * dx;
    T[i][0] = 10. * sin(pi * x / len);
  }
  save_stamp(0, 0);

  for (int k = 1; k <= M; ++k) {
    int cur = k & 1;
    int prev = cur ^ 1;
    #pragma omp parallel for
    for (int i = 1; i < N; ++i) {
      T[i][cur] = T[i - 1][prev] * K / 2 +
        T[i][prev] * (1 - K) +
        T[i + 1][prev] * K / 2;
    }
    T[0][cur] = left_val;
    T[N][cur] = T[N - 1][cur] - right_val;
    if (k % (M / 10) == 0) {
      save_stamp(cur, dt * k);
    }
  }

  double t1 = omp_get_wtime();
  cerr << "time: " << t1 - t0 << " s." << endl;
}

void out() {
  cout.setf(ios::fixed);
  cout.precision(5);
  cout << res_size << endl; // number of lines
  for (int i = 0; i < res_size; ++i) {
    cout << res_time[i] << endl;
    for (int j = 0; j <= N; ++j) {
      cout << j * dx << " ";
    }
    cout << endl;
    for (int j = 0; j <= N; ++j) {
      cout << res[j][i] << " ";
    }
    cout << endl;
  }
}

int main(int argc, char **argv) {
  freopen("output.txt", "w", stdout);

  dbg(K);
  dbg(dt);
  dbg(dx);
  dbg(right_val);
  dbg(omp_get_max_threads());

  process();
  out();
  return 0;
}
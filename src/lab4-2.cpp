#include <bits/stdc++.h>
#include <omp.h>

using namespace std;

struct vec_t {
  double x, y;
  vec_t() : x(0), y(0) {}
  vec_t(double x, double y) : x(x), y(y) {}
  vec_t operator - (const vec_t &b) const {
    return vec_t(x - b.x, y - b.y);
  }
  vec_t operator + (const vec_t &b) const {
    return vec_t(x + b.x, y + b.y);
  }
  vec_t operator / (double val) const {
    return vec_t(x / val, y / val);
  }
  double cross(const vec_t &b) const {
    return x * b.y - y * b.x;
  }
  vec_t rotate(double ang) const {
    return vec_t(+cos(ang) * x + sin(ang) * y,
                 -sin(ang) * x + cos(ang) * y);
  }
};

const int N = 1000000;
vec_t points[N];

const int K = 4;
double prob[K] = { 0.01, 0.85, 0.07, 0.07 };
double a[K] = { 0.00, 0.85, 0.20, -0.15 };
double b[K] = { 0.00, 0.04, -0.26, 0.28 };
double c[K] = { 0.00, -0.04, 0.23, 0.26 };
double d[K] = { 0.16, 0.85, 0.22, 0.24 };
double e[K] = { 0.00, 0.00, 0.00, 0.00 };
double f[K] = { 0.00, 1.60, 1.60, 0.44 };

random_device rd;
mt19937 mt(rd());
uniform_real_distribution<double> rnd(0, 1);

int next_rand() {
  double x = rnd(mt);
  for (int i = 0; i < K; ++i) {
    if (x <= prob[i]) {
      return i;
    }
  }
  return -1;
}

vec_t transform(const vec_t &v) {
  int i = next_rand();
  double x = a[i] * v.x + b[i] * v.y + e[i];
  double y = c[i] * v.x + d[i] * v.y + f[i];
  return vec_t(x, y);
}

void process() {
  partial_sum(prob, prob + K, prob);
  points[0] = vec_t(0, 0);
  for (int i = 1; i < N; ++i) {
    points[i] = transform(points[i - 1]);
  }
}

void out(const char *path) {
  ofstream fout(path);
  for (int i = 0; i < N; ++i) {
    fout << points[i].x << " ";
  }
  fout << endl;
  for (int i = 0; i < N; ++i) {
    fout << points[i].y << " ";
  }
  fout << endl;
  fout.close();
}

int main() {
  double t0 = omp_get_wtime();
  process();
  double t1 = omp_get_wtime();
  cerr << "time = " << t1 - t0 << " s" << endl;
  out("output.txt");
  return 0;
}
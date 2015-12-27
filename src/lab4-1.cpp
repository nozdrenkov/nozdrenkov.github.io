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

random_device rd;
mt19937 mt(rd());

const double pi = acos(-1);
const double side = 1;

const vec_t a(0, 0);
const vec_t b(side, 0);
const vec_t c = (b - a).rotate(-pi / 3);

const vec_t & triangle_point(int id) {
  assert(0 <= id && id < 3);
  if (id == 0) return (a + b) / 2;
  if (id == 1) return (b + c) / 2;
  if (id == 2) return (c + a) / 2;
}

vec_t random_point() {
  uniform_real_distribution<double> rnd(0, 1);
  double h = c.y;
  double x = rnd(mt) * side / 2;
  double y = rnd(mt) * h;
  vec_t p(x, y);
  if ((p - a).cross(c - a) < 0) {
    p.x += side / 2;
    p.y = -p.y + h;
  }
  return p;
}

const int N = 100000;
vec_t points[N];

void process() {
  uniform_int_distribution<int> rnd(0, 2);
  points[0] = a;
  points[1] = b;
  points[2] = c;
  points[3] = random_point();
  for (int i = 4; i < N; ++i) {
    int id = rnd(mt);
    points[i] = (points[i - 1] + triangle_point(id)) / 2;
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
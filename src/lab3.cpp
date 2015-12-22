#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <random>

#define dbg(x) do { cerr << #x << " = " << x << endl; } while(0)

using namespace std;

// random
namespace random {
  random_device rd;
  mt19937 mt(rd());
  uniform_real_distribution<double> dist(0.0, 1.0);

  double next() {
    return dist(mt);
  }

  double next(double l, double r) {
    return l + next() * (r - l);
  }

  double next(double a, double b, double c, double d) {
    double len1 = b - a;
    double len2 = d - c;
    double cur = next() * (len1 + len2);
    if (cur <= len1) {
      return a + cur;
    } else {
      return c + cur - len1;
    }
  }
}

// input constraints
const int N = 100000;
const int K = 10;
const double initial_energey = 3.5 * 1000;
const double width = 0.5;
const double gamma = 1e5;
const double alpha = 1e5;
const double min_energy = 500;
const double pi = acos(-1);
const double elastic_scattering_prob = 0.8;
const double huge_angle_prob = 0.25;

const int energy_blocks = 30;
int transmitted_with_energey[energy_blocks];

// floating point operations
const double eps = 1e-9;

bool double_less(double x, double y) {
  return x < y - eps;
}
bool double_equal(double x, double y) {
  return abs(x - y) <= 2 * eps;
}
bool double_less_equal(double x, double y) {
  return double_less(x, y) || double_equal(x, y);
}

double to_rad(double x) {
  return pi * x / 180;
}

struct Particle {
  double x, y, energy, angle;
  Particle()
    : x(0), y(0), energy(initial_energey), angle(0) {}
  bool has_energy() const {
    return energy >= 500;
  }
  bool reflected() const {
    return double_less(x, 0);
  }
  bool absorpted() const {
    return double_less_equal(0, x) && double_less_equal(x, width);
  }
  bool transmitted() const {
    return double_less(width, x);
  }
};

double new_angle() {
  double angle = 0;
  if (random::next() <= elastic_scattering_prob) {
    if (random::next() <= huge_angle_prob) {
      angle = random::next(-180, -10, +10, +180);
    } else {
      angle = random::next(-10, -2, +2, +10);
    }
  } else {
    angle = random::next(-2, +2);
  }
  return to_rad(angle);
}

void update_pos(Particle &p) {
  double len = -p.energy / gamma * log(random::next());
  p.angle += new_angle();
  p.energy -= sqrt(2 * alpha * len);
  p.x += len * cos(p.angle);
  p.y += len * sin(p.angle);
}

Particle a[N];
bool removed[N];

int active = N;
int reflected = 0;
int absorpted = 0;
int transmitted = 0;

typedef pair<double, double> Pt;
vector<Pt> trajectory[K];

void process() {
  for (int i = 0; i < K; ++i) {
    trajectory[i].emplace_back(0, 0);
  }
  int max_len = 0;
  while (active) {
    ++max_len;
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
      if (!removed[i]) {
        update_pos(a[i]);
        
        bool remove = false;
        if (!a[i].has_energy() && a[i].absorpted()) {
          absorpted++;
          remove = true;
        } else if (a[i].reflected()) {
          reflected++;
          remove = true;
        } else if (a[i].transmitted()) {
          transmitted++;
          int energy_block = int(a[i].energy / initial_energey * energy_blocks);
          transmitted_with_energey[energy_block]++;
          remove = true;
        }
        
        if (i < K) {
          trajectory[i].emplace_back(a[i].x, a[i].y);
        }

        if (remove) {
          active--;
          removed[i] = true;
        }
      }
    }
  }
  dbg(max_len);
}

void out() {
  cerr << "reflected = " << reflected * 100. / N << " %" << endl;
  cerr << "absorpted = " << absorpted * 100. / N << " %" << endl;
  cerr << "transmitted = " << transmitted * 100. / N << " %" << endl;
  cout.setf(ios::fixed);
  cout.precision(5);

  cout << "Transmitted:" << endl;
  for (int cnt : transmitted_with_energey) {
    cout << cnt << " ";
  }
  cout << endl << endl;

  cout << "Trajectories:" << endl;
  cout << K << endl;
  for (int i = 0; i < K; ++i) {
    for (const auto &p : trajectory[i]) {
      cout << p.first << " ";
    }
    cout << endl;
    for (const auto &p : trajectory[i]) {
      cout << p.second << " ";
    }
    cout << endl;
  }
}

int main() {
  freopen("output.txt", "w", stdout);
  process();
  out();
  return 0;
}
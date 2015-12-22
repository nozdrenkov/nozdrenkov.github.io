#include <bits/stdc++.h>
#include <omp.h>

using namespace std;

const int allPoints = 1e5;
const int nThreads = 4;
const int nPoints = allPoints / nThreads;
const int nBlocks = 50;

uniform_real_distribution<double> dist(0, 1);
mt19937 generator[nThreads];   // mersenne twister engine

int thread_count[nThreads][nBlocks];
int result_count[nBlocks];

void process() {
  if (nThreads * nPoints != allPoints) {
    cerr << "Choose allPoints divisible by nThreads" << endl;
    exit(0);
  }

  double t0 = omp_get_wtime();

  // init
  omp_set_num_threads(nThreads);
  for (int i = 0; i < nThreads; ++i) {
    generator[i].seed(random_device()());
  }

  // parallel generation
  #pragma omp parallel for
  for (int thread = 0; thread < nThreads; ++thread) {
    for (int i = 0; i < nPoints; ++i) {
      double val = dist(generator[thread]);
      int block = int(val * nBlocks);
      assert(block >= 0 && "impossible_zero!");
      assert(block < nBlocks && "impossoble_one!");
      thread_count[thread][block]++;
    }
  }
  #pragma omp parallel for
  for (int block = 0; block < nBlocks; ++block) {
    for (int thread = 0; thread < nThreads; ++thread) {
      result_count[block] += thread_count[thread][block];
    }
  }

  double t1 = omp_get_wtime();
  cerr << "time = " << t1 - t0 << " s" << endl;

  // check
  int points_cnt = accumulate(result_count, result_count + nBlocks, 0);
  if (points_cnt != allPoints) {
    cout << "error!" << endl;
  } else {
    cout << "ok!" << endl;
  }
}

void out() {
  ofstream fout("output.txt");
  fout.setf(ios::fixed);
  fout.precision(8);
  double criteria = 0;
  for (int cnt : result_count) {
    criteria += pow(allPoints - double(nBlocks) * cnt, 2) / allPoints;
    fout << 100 * double(cnt) / (allPoints) << " ";
  }
  cout << endl;
  cerr << "criteria = " << criteria << endl;
  fout.close();
}

int main() {
  process();
  out();
  return 0;
}
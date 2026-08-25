#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  const size_t n = keys.size();

  // ---- Precompute: park all keys & values in SRAM; transpose each key so
  // that keys[j] is (d x 1), ready to multiply with query rows. ----------
  for (size_t j = 0; j < n; ++j) {
    gpu_sim.MoveMatrixToSharedMem(keys[j]);
    gpu_sim.MoveMatrixToSharedMem(values[j]);
  }
  for (size_t j = 0; j < n; ++j) {
    gpu_sim.Transpose(keys[j], kInSharedMemory);
  }

  for (size_t i = 0; i < n; ++i) {
    Matrix *Q = rater.GetNextQuery();
    const size_t m = i + 1; // number of key/value groups in this round
    gpu_sim.MoveMatrixToSharedMem(Q);

    // O[r][c] = ( sum_j exp(s_j[r]) * V_j[c] ) / ( sum_j exp(s_j[r]) )
    // where s_j[r] = dot(Q_r, K_j).  Accumulate numerator N (m x d) and
    // per-row denominator R (m x 1) column by column, avoiding a big
    // softmax (m x m) @ V matmul.
    Matrix *N = matrix_memory_allocator.Allocate("N_init");
    gpu_sim.MatSub(Q, Q, N); // zero (m x d) accumulator

    Matrix *R = nullptr;
    for (size_t j = 0; j < m; ++j) {
      Matrix *s = matrix_memory_allocator.Allocate("s");
      gpu_sim.MatMul(Q, keys[j], s);       // m x 1 : scores vs key j
      Matrix *e = matrix_memory_allocator.Allocate("e");
      gpu_sim.MatExp(s, e);                // m x 1 : exp(scores)
      if (j == 0) {
        R = matrix_memory_allocator.Allocate("R");
        gpu_sim.Copy(e, R, kInSharedMemory);
      } else {
        Matrix *Rn = matrix_memory_allocator.Allocate("Rn");
        gpu_sim.MatAdd(R, e, Rn);
        gpu_sim.ReleaseMatrix(R);
        R = Rn;
      }
      Matrix *t = matrix_memory_allocator.Allocate("t");
      gpu_sim.MatMul(e, values[j], t);     // m x d : e_j outer V_j
      Matrix *Nn = matrix_memory_allocator.Allocate("Nn");
      gpu_sim.MatAdd(N, t, Nn);
      gpu_sim.ReleaseMatrix(N);
      gpu_sim.ReleaseMatrix(t);
      gpu_sim.ReleaseMatrix(s);
      gpu_sim.ReleaseMatrix(e);
      N = Nn;
    }

    // Divide each row of N by the corresponding element of R.
    Matrix *answer = matrix_memory_allocator.Allocate("answer");
    for (size_t r = 0; r < m; ++r) {
      Matrix *num = matrix_memory_allocator.Allocate("num");
      gpu_sim.GetRow(N, r, num, kInSharedMemory);   // 1 x d
      Matrix *den = matrix_memory_allocator.Allocate("den");
      gpu_sim.GetRow(R, r, den, kInSharedMemory);   // 1 x 1
      Matrix *outrow = matrix_memory_allocator.Allocate("outrow");
      gpu_sim.MatDiv(num, den, outrow);             // 1 x d
      if (r == 0) {
        gpu_sim.Copy(outrow, answer, kInSharedMemory);
      } else {
        Matrix *na = matrix_memory_allocator.Allocate("na");
        gpu_sim.Concat(answer, outrow, na, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(answer);
        answer = na;
      }
      gpu_sim.ReleaseMatrix(num);
      gpu_sim.ReleaseMatrix(den);
      gpu_sim.ReleaseMatrix(outrow);
    }

    gpu_sim.MoveMatrixToGpuHbm(answer);
    gpu_sim.ReleaseMatrix(N);
    gpu_sim.ReleaseMatrix(R);
    gpu_sim.ReleaseMatrix(Q);
    gpu_sim.Run(false, &matrix_memory_allocator);

    rater.CommitAnswer(*answer);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu

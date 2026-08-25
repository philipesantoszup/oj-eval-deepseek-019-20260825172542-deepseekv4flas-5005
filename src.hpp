#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  const size_t n = keys.size();
  // Park all keys and values in SRAM permanently; transpose keys -> (d x 1).
  for (size_t j = 0; j < n; ++j) {
    gpu_sim.MoveMatrixToSharedMem(keys[j]);
    gpu_sim.MoveMatrixToSharedMem(values[j]);
  }
  for (size_t j = 0; j < n; ++j) gpu_sim.Transpose(keys[j], kInSharedMemory);

  for (size_t i = 0; i < n; ++i) {
    Matrix *Q = rater.GetNextQuery();       // HBM, (m x d)
    const size_t m = i + 1;
    Matrix *answer = nullptr;               // answer built in HBM
    for (size_t r = 0; r < m; ++r) {
      // row of Q: extract from HBM, drop into SRAM
      Matrix *Qr = matrix_memory_allocator.Allocate("Qr");
      gpu_sim.GetRow(Q, r, Qr, kInGpuHbm);
      gpu_sim.MoveMatrixToSharedMem(Qr);    // (1 x d) in SRAM

      Matrix *num = matrix_memory_allocator.Allocate("num");
      gpu_sim.MatSub(Qr, Qr, num);          // zero (1 x d)
      Matrix *rowR = nullptr;               // (1 x 1) row denominator
      for (size_t j = 0; j < m; ++j) {
        Matrix *s = matrix_memory_allocator.Allocate("s");
        gpu_sim.MatMul(Qr, keys[j], s);     // 1 x 1 score
        Matrix *e = matrix_memory_allocator.Allocate("e");
        gpu_sim.MatExp(s, e);               // 1 x 1
        if (j == 0) { rowR = matrix_memory_allocator.Allocate("rowR"); gpu_sim.Copy(e, rowR, kInSharedMemory); }
        else { Matrix *r2 = matrix_memory_allocator.Allocate("r2"); gpu_sim.MatAdd(rowR, e, r2); gpu_sim.ReleaseMatrix(rowR); rowR = r2; }
        Matrix *t = matrix_memory_allocator.Allocate("t");
        gpu_sim.MatMul(e, values[j], t);    // (1 x 1) x (1 x d) -> (1 x d)
        Matrix *n2 = matrix_memory_allocator.Allocate("n2");
        gpu_sim.MatAdd(num, t, n2);
        gpu_sim.ReleaseMatrix(num);
        gpu_sim.ReleaseMatrix(t);
        gpu_sim.ReleaseMatrix(s);
        gpu_sim.ReleaseMatrix(e);
        num = n2;
      }
      Matrix *outrow = matrix_memory_allocator.Allocate("outrow");
      gpu_sim.MatDiv(num, rowR, outrow);    // (1 x d)
      gpu_sim.MoveMatrixToGpuHbm(outrow);   // -> HBM
      if (r == 0) {
        answer = matrix_memory_allocator.Allocate("answer");
        gpu_sim.Copy(outrow, answer, kInGpuHbm);
      } else {
        Matrix *na = matrix_memory_allocator.Allocate("na");
        gpu_sim.Concat(answer, outrow, na, 0, kInGpuHbm);
        gpu_sim.ReleaseMatrix(answer);
        answer = na;
      }
      gpu_sim.ReleaseMatrix(Qr);
      gpu_sim.ReleaseMatrix(num);
      gpu_sim.ReleaseMatrix(rowR);
      gpu_sim.ReleaseMatrix(outrow);
    }
    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*answer);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim, matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu

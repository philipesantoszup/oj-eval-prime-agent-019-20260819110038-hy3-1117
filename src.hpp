#pragma once
#include "simulator.hpp"
namespace sjtu {

// Helper: ensure a matrix is in SRAM. We must NOT re-issue a move for a
// matrix that is already in SRAM (the simulator would corrupt its memory
// accounting). So we check the current position first.
static inline void ensure_shared(GpuSimulator &gpu_sim, Matrix *m) {
  if (m->GetPosition() != Position::kInSharedMemory) {
    gpu_sim.MoveMatrixToSharedMem(m);
  }
}

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();  // Q, shape [(i+1), 512], in HBM
    const size_t n = i + 1;

    // ---- Pre-move all inputs we will need this round into SRAM up front.
    // These IO transfers overlap with the (much longer) score-matrix
    // matmul below, so they add essentially no extra wall-clock cycles.
    ensure_shared(gpu_sim, current_query);
    for (size_t j = 0; j < n; ++j) {
      ensure_shared(gpu_sim, keys[j]);
      ensure_shared(gpu_sim, values[j]);
    }

    // ---- Build score matrix S = Q * [K_0^T, K_1^T, ..., K_{n-1}^T]  (shape [n,n])
    Matrix *S = nullptr;
    for (size_t j = 0; j < n; ++j) {
      // Work on a transposed copy so the original key is not corrupted.
      Matrix *kt = matrix_memory_allocator.Allocate("kt");
      gpu_sim.Copy(keys[j], kt, Position::kInSharedMemory);
      gpu_sim.Transpose(kt, Position::kInSharedMemory);  // kt becomes [512,1]

      Matrix *col = matrix_memory_allocator.Allocate("col");
      gpu_sim.MatMul(current_query, kt, col);  // col = Q * Kt  -> [n,1]

      if (S == nullptr) {
        S = col;
      } else {
        Matrix *newS = matrix_memory_allocator.Allocate("S");
        gpu_sim.Concat(S, col, newS, 1, Position::kInSharedMemory);  // column concat
        gpu_sim.ReleaseMatrix(S);
        gpu_sim.ReleaseMatrix(col);
        S = newS;
      }
      gpu_sim.ReleaseMatrix(kt);
    }

    // Q is no longer needed for the rest of this round.
    gpu_sim.ReleaseMatrix(current_query);

    // ---- Softmax over rows of S.
    Matrix *E = matrix_memory_allocator.Allocate("E");
    gpu_sim.MatExp(S, E);
    gpu_sim.ReleaseMatrix(S);

    Matrix *softmax = nullptr;
    for (size_t r = 0; r < n; ++r) {
      Matrix *row = matrix_memory_allocator.Allocate("row");
      gpu_sim.GetRow(E, r, row, Position::kInSharedMemory);
      Matrix *s = matrix_memory_allocator.Allocate("s");
      gpu_sim.Sum(row, s);  // s = sum of row (1x1)
      Matrix *norm = matrix_memory_allocator.Allocate("norm");
      gpu_sim.MatDiv(row, s, norm);  // divide each element by the row sum
      gpu_sim.ReleaseMatrix(row);
      gpu_sim.ReleaseMatrix(s);

      if (softmax == nullptr) {
        softmax = norm;
      } else {
        Matrix *newSoft = matrix_memory_allocator.Allocate("soft");
        gpu_sim.Concat(softmax, norm, newSoft, 0,
                       Position::kInSharedMemory);  // row concat
        gpu_sim.ReleaseMatrix(softmax);
        gpu_sim.ReleaseMatrix(norm);
        softmax = newSoft;
      }
    }
    gpu_sim.ReleaseMatrix(E);

    // ---- Output = sum_j softmax[:,j] (outer) V_j   (shape [n,512])
    // Values were already moved to SRAM at the start of the round, so this
    // loop is pure computation (no IO stalls).
    Matrix *output = nullptr;
    for (size_t j = 0; j < n; ++j) {
      Matrix *scol = matrix_memory_allocator.Allocate("scol");
      gpu_sim.GetColumn(softmax, j, scol, Position::kInSharedMemory);  // [n,1]
      Matrix *contrib = matrix_memory_allocator.Allocate("contrib");
      gpu_sim.MatMul(scol, values[j], contrib);  // [n,1] x [1,512] -> [n,512]
      gpu_sim.ReleaseMatrix(scol);

      if (output == nullptr) {
        output = contrib;
      } else {
        Matrix *newOut = matrix_memory_allocator.Allocate("out");
        gpu_sim.MatAdd(output, contrib, newOut);
        gpu_sim.ReleaseMatrix(output);
        gpu_sim.ReleaseMatrix(contrib);
        output = newOut;
      }
    }
    gpu_sim.ReleaseMatrix(softmax);

    // Move the answer to HBM and commit.
    gpu_sim.MoveMatrixToGpuHbm(output);
    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*output);
    // output is released automatically by CommitAnswer.
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu

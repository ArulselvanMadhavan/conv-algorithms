#include <ATen/ATen.h>
#include <ATen/core/Formatting.h>
#include <ATen/ops/convolution.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/rand.h>
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/SmallVector.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <optional>
#include <torch/torch.h>
#include <vector>

float gen_rand() {
  return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

void fill_matrix(std::vector<float> &in, bool use_zero) {
  for (int i = 0; i < in.size(); i++) {
    in[i] = use_zero ? 0.0 : 1.0;
    // gen_rand();
  }
}

unsigned int numel(at::IntArrayRef in_shape) {
  unsigned int size = 1;
  for (int i = 0; i < in_shape.size(); i++) {
    size = size * in_shape[i];
  }
  return size;
}

void naive_conv2d(std::vector<float> &in, std::vector<float> &k,
                  std::vector<float> &out, at::IntArrayRef in_shape,
                  at::IntArrayRef k_shape, at::IntArrayRef o_shape) {
  int C = in_shape[0];
  int H = in_shape[1];
  int W = in_shape[2];
  int M = k_shape[0];
  int K = k_shape[1];

  for (int h = 0; h < H; h++) {
    for (int w = 0; w < W; w++) {
      for (int o = 0; o < M; o++) {
        float sum = 0;
        for (int x = 0; x < K; x++) {
          for (int y = 0; y < K; y++) {
            for (int i = 0; i < C; i++) {
              int row_idx = h + y;
              int col_idx = w + x;
              if ((row_idx >= H) || (row_idx < 0)) {
                continue;
              }
              if ((col_idx >= W) || (col_idx < 0)) {
                continue;
              }
              int in_idx = i * row_idx + col_idx;
              int k_idx = (o * x * y) + i;
              sum += in[in_idx] * k[k_idx];
            }
          }
        }
        out[(o * h) + w] = sum;
      }
    }
  }
}

int main(int argc, char *argv[]) {
  auto seed = static_cast<unsigned>(time(0));
  srand(seed);
  std::cout << "Running kn2row with seed:" << seed << "\n";

  int H = 3;
  int W = 3;
  int C = 1;

  int M = 1;
  int K = 3;
  at::IntArrayRef in_shape = {C, H, W};
  at::IntArrayRef k_shape = {M, K, K, C};
  at::IntArrayRef o_shape = {M, H, W};

  unsigned int in_numel = numel(in_shape);
  unsigned int k_numel = numel(k_shape);
  unsigned int o_numel = numel(o_shape);

  std::vector<float> inputs(in_numel);
  std::vector<float> kernels(k_numel);
  std::vector<float> outputs(o_numel);

  fill_matrix(inputs, false);
  fill_matrix(kernels, false);
  fill_matrix(outputs, true);
  auto in_at = at::from_blob(inputs.data(), in_shape);
  auto k_at = at::from_blob(kernels.data(), k_shape);
  naive_conv2d(inputs, kernels, outputs, in_shape, k_shape, o_shape);
  auto o_at = at::from_blob(outputs.data(), o_shape);
  at::print(in_at);
  at::print(k_at);
  at::print(o_at);
  k_at = at::permute(k_at, {0, 3, 1, 2});
  in_at = at::unsqueeze(in_at, 0);
  // at::IntArrayRef stride {1,1};
  // at::IntArrayRef padding {0, 0};
  // at::IntArrayRef dilation {1, 1};
  auto o_ref = at::convolution(in_at, k_at, c10::nullopt, {1, 1}, {0, 0},
                               {1, 1}, false, {0, 0}, 1);
  at::print(o_ref);
}

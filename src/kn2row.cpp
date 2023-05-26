#include <ATen/ATen.h>
#include <ATen/core/Formatting.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/rand.h>
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/SmallVector.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <torch/torch.h>
#include <vector>

float gen_rand() {
  return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

void fill_matrix(std::vector<float> &in) {
  for (int i = 0; i < in.size(); i++) {
    in[i] = gen_rand();
  }
}

unsigned int numel(at::IntArrayRef in_shape) {
  unsigned int size = 1;
  for(int i = 0; i < in_shape.size(); i++){
    size = size * in_shape[i];
  }
  return size;
}

int main(int argc, char *argv[]) {
  auto seed = static_cast<unsigned>(time(0));
  srand(seed);
  std::cout << "Running kn2row with seed:" << seed << "\n";

  int H = 25;
  int W = 25;
  int C = 3;

  int M = 20;
  int K = 3;
  at::IntArrayRef in_shape = {H, W, C};
  at::IntArrayRef k_shape = {M, C, K, K};
  at::IntArrayRef o_shape = {M, H, W};
  
  unsigned int in_numel = numel(in_shape);
  unsigned int k_numel = numel(k_shape);
  unsigned int o_numel = numel(o_shape);
  
  std::vector<float> inputs(in_numel);
  std::vector<float> kernels(k_numel);
  std::vector<float> outputs(o_numel);
  
  fill_matrix(inputs);
  fill_matrix(kernels);
  fill_matrix(outputs);
  auto in_at = at::from_blob(inputs.data(), in_shape);
  auto k_at = at::from_blob(kernels.data(), k_shape);
  auto o_at = at::from_blob(outputs.data(), o_shape);
}

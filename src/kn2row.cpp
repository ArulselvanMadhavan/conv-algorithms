#include <ATen/ATen.h>
#include <ATen/core/Formatting.h>
#include <ATen/ops/convolution.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/rand.h>
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/SmallVector.h>
#include <cmath>
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
    in[i] = use_zero ? 0.0 : gen_rand();
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
  int N = in_shape[0];
  int C = in_shape[1];
  int H = in_shape[2];
  int W = in_shape[3];
  int M = k_shape[0];
  int K = k_shape[2];
  for (int n = 0; n < N; n++) {
    for (int o = 0; o < M; o++) {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          float sum = 0;
          for (int c = 0; c < C; c++) {
            for (int x = 0; x < K; x++) {
              for (int y = 0; y < K; y++) {
                int k_idx = (o * C * K * K) + (c * K * K) + (x * K) + y;
                int h_idx = h + (x - K / 2);
                int w_idx = w + (y - K / 2);
                if ((h_idx >= H) || (h_idx < 0)) {
                  continue;
                }
                if ((w_idx >= W) || (w_idx < 0)) {
                  continue;
                }
                int in_idx =
                    (n * C * H * W) + (c * H * W) + (h_idx * W) + w_idx;
                sum += in[in_idx] * k[k_idx];
              }
            }
          }
          int out_idx = (o * H * W) + (h * H) + w;
          out[out_idx] = sum;
        }
      }
    }
  }
}

int main(int argc, char *argv[]) {
  auto seed = static_cast<unsigned>(time(0));
  srand(seed);
  std::cout << "Running kn2row with seed:" << seed << "\n";

  int N = 1;
  int H = 3;
  int W = 3;
  int C = 1;

  int M = 1;
  int K = 3;
  std::array<long int, 2> padding{1, 1}, stride{1, 1}, dilation{1, 1};
  std::array<long int, 4> in_shape{N, C, H, W};
  std::array<long int, 4> k_shape{M, C, K, K};
  std::array<long int, 4> o_shape = {N, M, H, W};

  unsigned int in_numel = numel(in_shape);
  unsigned int k_numel = numel(k_shape);
  unsigned int o_numel = numel(o_shape);

  std::vector<float> inputs(in_numel);
  std::vector<float> kernels(k_numel);
  std::vector<float> outputs(o_numel);

  fill_matrix(inputs, false);
  fill_matrix(kernels, false);
  fill_matrix(outputs, true);
  naive_conv2d(inputs, kernels, outputs, in_shape, k_shape, o_shape);

  // Torch ref
  auto in_at = at::from_blob(inputs.data(), in_shape);
  auto k_at = at::from_blob(kernels.data(), k_shape);
  auto o_at = at::from_blob(outputs.data(), o_shape);

  at::IntArrayRef pad_ref = at::ArrayRef(padding.begin(), padding.end());
  at::IntArrayRef str_ref = at::ArrayRef(stride.begin(), stride.end());
  at::IntArrayRef dil_ref = at::ArrayRef(stride.begin(), stride.end());
  auto o_ref = at::convolution(in_at, k_at, c10::nullopt, str_ref, pad_ref,
                               dil_ref, false, {0, 0}, 1);
  bool result =
      at::isclose(o_at, o_ref, 1e-3, 1e-3, true).all().item<uint8_t>();
  std::cout << "IsMatch:" << result << "\n";
}

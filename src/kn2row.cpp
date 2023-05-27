#include "argh.h"
#include <ATen/ATen.h>
#include <ATen/core/Formatting.h>
#include <ATen/ops/convolution.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/rand.h>
#include <algorithm>
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/SmallVector.h>
#include <cblas.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <optional>
#include <string>
#include <torch/torch.h>
#include <vector>

using namespace argh;

float gen_rand() {
  return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

void fill_matrix(std::vector<float> &in, bool use_zero) {
  for (unsigned long i = 0; i < in.size(); i++) {
    in[i] = use_zero ? 0.0 : gen_rand();
  }
}

unsigned int numel(at::IntArrayRef in_shape) {
  unsigned int size = 1;
  for (unsigned long i = 0; i < in_shape.size(); i++) {
    size = size * in_shape[i];
  }
  return size;
}

void conv2d(std::vector<float> &in, std::vector<float> &k,
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
                if ((h_idx >= H) || (h_idx < 0) || (w_idx >= W) ||
                    (w_idx < 0)) {
                  continue;
                }
                int in_idx =
                    (n * C * H * W) + (c * H * W) + (h_idx * W) + w_idx;
                sum += in[in_idx] * k[k_idx];
              }
            }
          }
          int out_idx = (n * M * H * W) + (o * H * W) + (h * H) + w;
          out[out_idx] = sum;
        }
      }
    }
  }
}

void swap(int src_idx, int dst_idx, size_t size, float *out_begin,
          float *temp) {
  std::memcpy(temp, out_begin + src_idx, size);
  // for (int i = 0; i < size; i++) {
  //   std::cout << "temp:" << temp[i] << "\n";
  // }
  std::memcpy(out_begin + src_idx, out_begin + dst_idx, size);
  std::memcpy(out_begin + dst_idx, temp, size);
}
void im2col_scan(std::vector<float> &in, std::vector<float> &k,
                 std::vector<float> &out, at::IntArrayRef in_shape,
                 at::IntArrayRef k_shape, at::IntArrayRef o_shape) {
  int N = in_shape[0];
  int C = in_shape[1];
  int H = in_shape[2];
  int W = in_shape[3];
  int M = k_shape[0];
  int K = k_shape[2];
  std::array<long int, 6> p_shape = {C, K, K, N, H, W};
  at::IntArrayRef p_ref = at::ArrayRef<int64_t>(p_shape.begin(), p_shape.end());
  std::vector<float> patches(numel(p_ref));
  fill_matrix(patches, true);
  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      for (int kh = 0; kh < K; kh++) {
        for (int kw = 0; kw < K; kw++) {
          for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
              int pch_idx = (c * K * K * N * H * W) + (kh * K * N * H * W) +
                            (kw * N * H * W) + (n * H * W) + (h * W) + w;
              int h_idx = h + (kh - K / 2);
              int w_idx = w + (kw - K / 2);
              if ((h_idx >= H) || (h_idx < 0) || (w_idx >= W) || (w_idx < 0)) {
                continue;
              }
              int in_idx = (n * C * H * W) + (c * H * W) + (h_idx * W) + w_idx;
              patches[pch_idx] = in[in_idx];
            }
          }
        }
      }
    }
  }
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N * H * W,
              C * K * K, 1.0, k.data(), C * K * K, patches.data(), N * H * W,
              1., out.data(), N * H * W);

  // Permute M N H W -> N M H W
  int X = std::min(M, N);
  int hw = H * W;
  std::vector<float> temp(hw);
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      int src_idx = (m * N) + n;
      int dst_idx = (n * M) + m;
      if (src_idx == dst_idx) {
        continue;
      }
      if ((m < X) && (n < X)) {
        if (m < n) { // Treat upper diag as src and lower diag as dest
          std::cout << src_idx << "|" << dst_idx << "\n";
          swap(src_idx * hw, dst_idx * hw, sizeof(float) * hw, out.data(),
               temp.data());
        }
      } else {
        std::cout << "Non-squ:" << src_idx << "|" << dst_idx << "\n";
        swap(src_idx * hw, dst_idx * hw, sizeof(float) * hw, out.data(),
             temp.data());
      }
    }
  }
  // auto o_at = at::from_blob(out.data(), o_shape);
  // o_at = o_at.permute({1, 0, 2, 3});
  // at::print(o_at);
}

enum ConvAlg { vanilla, im2col };

static std::map<std::string, ConvAlg> const table = {
    {"vanilla", ConvAlg::vanilla}, {"im2col", ConvAlg::im2col}};
static const std::string convAlgParam("conv-alg");

int main(int argc, char *argv[]) {
  argh::parser cmdl;
  cmdl.add_param(convAlgParam);
  cmdl.add_param({"-N", "--batch-size", "-H", "--height", "-W", "--width", "-C",
                  "--channels", "-M", "--out-channels", "-K", "--kernel-size"});
  cmdl.parse(argc, argv, argh::parser::PREFER_PARAM_FOR_UNREG_OPTION);

  // Handle conv alg
  auto conv_alg = cmdl(convAlgParam).str();
  auto it = table.find(conv_alg);
  ConvAlg convAlg;
  if (it != table.end()) {
    convAlg = it->second;
  } else {
    std::cout << "Unsupported conv alg:" << conv_alg << "\n";
    return -1;
  }

  // Handle int params
  int N = std::atoi(cmdl("N").str().c_str());
  int H = std::atoi(cmdl("H").str().c_str());
  int W = std::atoi(cmdl("W").str().c_str());
  int C = std::atoi(cmdl("C").str().c_str());
  int M = std::atoi(cmdl("M").str().c_str());
  int K = std::atoi(cmdl("K").str().c_str());

  // Seed
  auto seed = static_cast<unsigned>(time(0));
  seed = 41;
  srand(seed);
  std::cout << "Running kn2row with seed:" << seed << "\n";

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

  switch (convAlg) {
  case ConvAlg::vanilla:
    conv2d(inputs, kernels, outputs, in_shape, k_shape, o_shape);
    break;
  case ConvAlg::im2col:
    im2col_scan(inputs, kernels, outputs, in_shape, k_shape, o_shape);
    break;
  };

  // Torch ref
  auto in_at = at::from_blob(inputs.data(), in_shape);
  auto k_at = at::from_blob(kernels.data(), k_shape);
  auto o_at = at::from_blob(outputs.data(), o_shape);

  at::IntArrayRef pad_ref =
      at::ArrayRef<int64_t>(padding.begin(), padding.end());
  at::IntArrayRef str_ref = at::ArrayRef<int64_t>(stride.begin(), stride.end());
  at::IntArrayRef dil_ref =
      at::ArrayRef<int64_t>(dilation.begin(), dilation.end());
  auto o_ref = at::convolution(in_at, k_at, c10::nullopt, str_ref, pad_ref,
                               dil_ref, false, {0, 0}, 1);
  bool result =
      at::isclose(o_at, o_ref, 1e-3, 1e-3, true).all().item<uint8_t>();
  if (!result) {
    at::print(o_at);
    at::print(o_ref);
  }
  return result ? 0 : -1;
}

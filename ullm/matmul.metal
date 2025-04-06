// matmul.metal
#include <metal_stdlib>
using namespace metal;

kernel void matmul_vec(const device float* W [[ buffer(0) ]],
                       const device float* x [[ buffer(1) ]],
                       device float* y [[ buffer(2) ]],
                       constant uint& n [[ buffer(3) ]],
                       uint id [[ thread_position_in_grid ]]) {
    float sum = 0.0;
    for (uint j = 0; j < n; ++j) {
        sum += W[id * n + j] * x[j];
    }
    y[id] = sum;
}
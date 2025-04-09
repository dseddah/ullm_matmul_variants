#include <metal_stdlib>
using namespace metal;

kernel void matmul_vec(const device float* W [[ buffer(0) ]],
                       const device float* x [[ buffer(1) ]],
                       device float* y [[ buffer(2) ]],
                       constant uint& n [[ buffer(3) ]],
                       constant uint& d [[ buffer(4) ]],
                       uint id [[ thread_position_in_grid ]]) {
    if (id >= d) return;

    float sum = 0.0f;
    for (uint j = 0; j < n; ++j) {
        sum += W[id * n + j] * x[j]; // W is (d x n), row-major
    }
    y[id] = sum;
}

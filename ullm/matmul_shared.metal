#include <metal_stdlib>
using namespace metal;

kernel void matmul_shader_shared_memory(
    const device float* W [[ buffer(0) ]],
    const device float* x [[ buffer(1) ]],
    device float* y [[ buffer(2) ]],
    constant uint& n [[ buffer(3) ]],
    constant uint& d [[ buffer(4) ]],
    uint id [[ thread_position_in_grid ]],
    uint tid [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]],
    threadgroup float* x_shared [[ threadgroup(0) ]]
) {
    if (id >= d) return; // 💥 Don't compute if output index is out-of-bounds

    // Load x into fast threadgroup memory
    for (uint j = tid; j < n; j += tg_size) {
        x_shared[j] = x[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum = 0.0f;
    for (uint j = 0; j < n; ++j) {
        sum += W[id * n + j] * x_shared[j];
    }

    y[id] = sum;
}

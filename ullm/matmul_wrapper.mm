// matmul_wrapper.mm
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

extern "C" void metal_matmul(float* x, float* w, float* y, int n, int d) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLLibrary> library = [device newLibraryWithFile:@"matmul.metallib" error:nil];
    id<MTLFunction> function = [library newFunctionWithName:@"matmul_vec"];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:nil];
    
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

    id<MTLBuffer> wBuf = [device newBufferWithBytes:w length:n*d*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> xBuf = [device newBufferWithBytes:x length:n*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> yBuf = [device newBufferWithLength:d*sizeof(float) options:MTLResourceStorageModeShared];

    uint32_t nn = (uint32_t)n;
    id<MTLBuffer> nBuf = [device newBufferWithBytes:&nn length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:wBuf offset:0 atIndex:0];
    [encoder setBuffer:xBuf offset:0 atIndex:1];
    [encoder setBuffer:yBuf offset:0 atIndex:2];
    [encoder setBuffer:nBuf offset:0 atIndex:3];

    MTLSize gridSize = MTLSizeMake(d, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(1, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    memcpy(y, [yBuf contents], d * sizeof(float));
}

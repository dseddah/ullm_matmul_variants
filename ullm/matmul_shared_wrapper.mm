#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
/*
#define ENABLE_MATMUL_LOGS 1
#if ENABLE_MATMUL_LOGS
    //NSLog(@"🚀 Dispatching Metal matmul: n=%d, d=%d", n, d);
#endif
*/

extern "C" void matmul_shader_shared_memory(float* x, float* w, float* y, int n, int d) {
    if (!x || !w || !y || n <= 0 || d <= 0) {
        //NSLog(@"❌ Invalid inputs: x=%p, w=%p, y=%p, n=%d, d=%d", x, w, y, n, d);
        return;
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        //NSLog(@"❌ Failed to get Metal device");
        return;
    }

    NSError* err = nil;
    NSString* libPath = @"out/matmul_shared.metallib";
    id<MTLLibrary> library = [device newLibraryWithFile:libPath error:&err];
    if (!library) {
        //NSLog(@"❌ Failed to load Metal library: %@", err);
        return;
    }

    id<MTLFunction> function = [library newFunctionWithName:@"matmul_shader_shared_memory"];
    if (!function) {
        //NSLog(@"❌ Failed to get Metal function");
        return;
    }

    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&err];
    if (!pipeline) {
        //NSLog(@"❌ Failed to create compute pipeline: %@", err);
        return;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

    id<MTLBuffer> wBuf = [device newBufferWithBytes:w length:n * d * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> xBuf = [device newBufferWithBytes:x length:n * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> yBuf = [device newBufferWithLength:d * sizeof(float) options:MTLResourceStorageModeShared];
    uint32_t nn = (uint32_t)n;
    uint32_t dd = (uint32_t)d;
    id<MTLBuffer> nBuf = [device newBufferWithBytes:&nn length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> dBuf = [device newBufferWithBytes:&dd length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

    if (!wBuf || !xBuf || !yBuf || !nBuf || !dBuf) {
        //NSLog(@"❌ Failed to create Metal buffers");
        return;
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:wBuf offset:0 atIndex:0];
    [encoder setBuffer:xBuf offset:0 atIndex:1];
    [encoder setBuffer:yBuf offset:0 atIndex:2];
    [encoder setBuffer:nBuf offset:0 atIndex:3];
    [encoder setBuffer:dBuf offset:0 atIndex:4];

    NSUInteger sharedSize = n * sizeof(float);
    if (sharedSize > 32768) {
        //NSLog(@"⚠️ Threadgroup memory too big: %lu bytes", sharedSize);
        return;
    }
    [encoder setThreadgroupMemoryLength:sharedSize atIndex:0];

    MTLSize gridSize = MTLSizeMake(d, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(MIN(d, 64), 1, 1);
    //NSLog(@"🚀 Dispatching Metal matmul: n=%d, d=%d", n, d);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    void* result = [yBuf contents];
    if (!result) {
        //NSLog(@"❌ Failed to get contents from yBuf — kernel likely failed (n=%d, d=%d)", n, d);
        return;
    }

    memcpy(y, result, d * sizeof(float));
}

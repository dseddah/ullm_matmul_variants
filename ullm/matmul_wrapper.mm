#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#define ENABLE_MATMUL_LOGS 0	

extern "C" void matmul_C(float* y, float* x, float* w, int n, int d);

extern "C" void metal_matmul(float* x, float* w, float* y, int n, int d) {
    if (!x || !w || !y || n <= 0 || d <= 0) {
    #if ENABLE_MATMUL_LOGS
        NSLog(@"❌ Invalid input to metal_matmul: x=%p w=%p y=%p n=%d d=%d", x, w, y, n, d);
    #endif
        return;
    }

    // Fallback to CPU if dimensions are too large
    if (d > 32768 || n > 32768) {
    #if  ENABLE_MATMUL_LOGS
        NSLog(@"⚠️ Falling back to CPU matmul for n=%d d=%d", n, d);
    #endif
        matmul_C(y, x, w, n, d);
        return;
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
    #if ENABLE_MATMUL_LOGS
        NSLog(@"❌ Failed to create Metal device");
    #endif
        return;
    }

    NSError* err = nil;
    NSString* libPath = @"out/matmul.metallib";

#if ENABLE_MATMUL_LOGS
    NSLog(@"📦 Loading Metal shader from: %@", libPath);
#endif

    id<MTLLibrary> library = [device newLibraryWithFile:libPath error:&err];
    if (!library) {
    #if ENABLE_MATMUL_LOGS
        NSLog(@"❌ Failed to load Metal library: %@", err);
    #endif
        matmul_C(y, x, w, n, d);
        return;
    }

    id<MTLFunction> function = [library newFunctionWithName:@"matmul_vec"];
    if (!function) {
    #if ENABLE_MATMUL_LOGS
        NSLog(@"❌ Failed to find Metal function 'matmul_vec'");
    #endif
        matmul_C(y, x, w, n, d);
        return;
    }

    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&err];
    if (!pipeline) {
    #if ENABLE_MATMUL_LOGS
        NSLog(@"❌ Failed to create pipeline: %@", err);
    #endif
        matmul_C(y, x, w, n, d);
        return;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

#if ENABLE_MATMUL_LOGS
    NSLog(@"🚀 Dispatching Metal matmul: n=%d, d=%d", n, d);
    NSLog(@"    grid=%d, threads/group=%d", d, MIN(64, d));
    NSLog(@"    x: %p, w: %p, y: %p", x, w, y);
#endif

    size_t wSize = (size_t)n * d * sizeof(float);
    size_t xSize = (size_t)n * sizeof(float);
    size_t ySize = (size_t)d * sizeof(float);

    id<MTLBuffer> wBuf = [device newBufferWithBytes:w length:wSize options:MTLResourceStorageModeShared];
    id<MTLBuffer> xBuf = [device newBufferWithBytes:x length:xSize options:MTLResourceStorageModeShared];
    id<MTLBuffer> yBuf = [device newBufferWithLength:ySize options:MTLResourceStorageModeShared];
    id<MTLBuffer> nBuf = [device newBufferWithBytes:&n length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> dBuf = [device newBufferWithBytes:&d length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

    if (!wBuf || !xBuf || !yBuf || !nBuf || !dBuf) {
    #if ENABLE_MATMUL_LOGS
        NSLog(@"❌ Failed to allocate one or more Metal buffers");
    #endif
        matmul_C(y, x, w, n, d);
        return;
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:wBuf offset:0 atIndex:0];
    [encoder setBuffer:xBuf offset:0 atIndex:1];
    [encoder setBuffer:yBuf offset:0 atIndex:2];
    [encoder setBuffer:nBuf offset:0 atIndex:3];
    [encoder setBuffer:dBuf offset:0 atIndex:4];

    MTLSize gridSize = MTLSizeMake(d, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(MIN(64, d), 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    
 	
 	void* result = [yBuf contents];
    if (!result) {
    #if ENABLE_MATMUL_LOGS
        NSLog(@"❌ Failed to read result buffer");
    #endif
        return;
    }

	memcpy(y, result, ySize);
    //memcpy(y, [yBuf contents], ySize);
}

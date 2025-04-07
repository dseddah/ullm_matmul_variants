#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

extern "C" void matmul_C(float* x, float* w, float* y, int n, int d);


extern "C" void metal_matmul(float* x, float* w, float* y, int n, int d) {
    if (!x || !w || !y || n <= 0 || d <= 0) return;
	if (d > 32768) {
    	NSLog(@"⚠️ Falling back to CPU matmul for d=%d", d);
    	matmul_C(y, x, w, n, d);
    	return;
	}


    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return;

    NSError* err = nil;
    NSString* libPath = @"out/matmul.metallib";
	#if 0
		NSLog(@"📦 Loading Metal shader from: %@", libPath);  // ✅ add this!
	#endif
	id<MTLLibrary> library = [device newLibraryWithFile:libPath error:&err];
    if (!library) return;

    id<MTLFunction> function = [library newFunctionWithName:@"matmul_vec"];
    if (!function) return;

    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&err];
    if (!pipeline) return;

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    
    #if 1

	
		if (!x || !w || !y) {
		NSLog(@"🧪 metal_matmul(n=%d, d=%d)", n, d);
		NSLog(@"    x: %p  w: %p  y: %p", x, w, y);
			NSLog(@"❌ One or more input pointers are null.");
			return;
		}
		
		if (n <= 0 || d <= 0 || n > 4096 || d > 4096) {
			NSLog(@"🧪 metal_matmul(n=%d, d=%d)", n, d);
			NSLog(@"    x: %p  w: %p  y: %p", x, w, y);
			NSLog(@"❌ Invalid dimensions: n=%d d=%d", n, d);
			return;
		} 
		
	#endif

    id<MTLBuffer> wBuf = [device newBufferWithBytes:w length:n * d * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> xBuf = [device newBufferWithBytes:x length:n * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> yBuf = [device newBufferWithLength:d * sizeof(float) options:MTLResourceStorageModeShared];
    uint32_t nn = (uint32_t)n;
    uint32_t dd = (uint32_t)d;
    id<MTLBuffer> nBuf = [device newBufferWithBytes:&nn length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> dBuf = [device newBufferWithBytes:&dd length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

    if (!wBuf || !xBuf || !yBuf || !nBuf || !dBuf) return;

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:wBuf offset:0 atIndex:0];
    [encoder setBuffer:xBuf offset:0 atIndex:1];
    [encoder setBuffer:yBuf offset:0 atIndex:2];
    [encoder setBuffer:nBuf offset:0 atIndex:3];
    [encoder setBuffer:dBuf offset:0 atIndex:4];

    MTLSize gridSize = MTLSizeMake(d, 1, 1);
	MTLSize threadgroupSize = MTLSizeMake(MIN(64, d), 1, 1); // ✅ fast  was 1,1,1

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    void* result = [yBuf contents];
    if (!result) return;

    memcpy(y, result, d * sizeof(float));
}

#import <Metal/Metal.h>
#import <Foundation/Foundation.h> // Import for NSString and NSLog
#import <CoreGraphics/CoreGraphics.h>

int main() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device) {
        NSLog(@"✅ Metal is available: %@", [device name]);
    } else {
        NSLog(@"❌ Metal is not available on this session.");
    }
    return 0;
}
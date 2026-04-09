#include <flash_attention.h>
#include <iostream>
#include <cl_context.h>

int main() {
    std::cout << "Test program" << std::endl;
    
    // Try to call the function to see if it links
    // This is just a test call, not a real test
    _FP16* dummy = nullptr;
    nntrainer::flash_attention_fp16_cl(dummy, dummy, dummy, dummy, 0, 0, 0, 0, 0, 0, 0.0f);
    
    return 0;
}

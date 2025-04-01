#include <sycl/sycl.hpp>
#include <iostream>
#include <filesystem>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void sobel_edge_detection(const unsigned char* input, unsigned char* output, int width, int height, int channels) {
    try {
        sycl::queue q(sycl::default_selector_v);
        std::cout << "Running on: " 
                 << q.get_device().get_info<sycl::info::device::name>() << "\n";

        // Create buffers
        sycl::buffer<unsigned char, 1> input_buffer(input, sycl::range(width * height * channels));
        sycl::buffer<unsigned char, 1> output_buffer(output, sycl::range(width * height * channels));

        q.submit([&](sycl::handler& h) {
            auto in = input_buffer.get_access<sycl::access::mode::read>(h);
            auto out = output_buffer.get_access<sycl::access::mode::write>(h);

            h.parallel_for(sycl::range<2>(height-2, width-2), [=](sycl::id<2> idx) {
                int y = idx[0] + 1;
                int x = idx[1] + 1;

                for (int c = 0; c < channels; c++) {
                    // Calculate indices
                    const int stride = width * channels;
                    int i00 = (y-1)*stride + (x-1)*channels + c;
                    int i10 = y*stride     + (x-1)*channels + c;
                    int i20 = (y+1)*stride + (x-1)*channels + c;
                    
                    // Sobel X kernel
                    float gx = -1.0f * in[i00] + 1.0f * in[i00 + 2*channels]
                               -2.0f * in[i10] + 2.0f * in[i10 + 2*channels]
                               -1.0f * in[i20] + 1.0f * in[i20 + 2*channels];

                    // Sobel Y kernel
                    float gy = -1.0f * in[i00] - 2.0f * in[i00 + channels] - 1.0f * in[i00 + 2*channels]
                               +1.0f * in[i20] + 2.0f * in[i20 + channels] + 1.0f * in[i20 + 2*channels];

                    // Gradient magnitude
                    float mag = sycl::sqrt(gx*gx + gy*gy);
                    out[y*stride + x*channels + c] = static_cast<unsigned char>(sycl::clamp(255.0f - mag, 0.0f, 255.0f));
                }
            });
        }).wait();
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL Exception: " << e.what() << std::endl;
        throw;
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " input.jpg output.png\n";
        return 1;
    }

    // Load image
    int width, height, channels;
    unsigned char* input = stbi_load(argv[1], &width, &height, &channels, 0);
    if (!input) {
        std::cerr << "Failed to load image: " << stbi_failure_reason() << std::endl;
        return 1;
    }
    std::cout << "Loaded: " << width << "x" << height << " (" << channels << " channels)\n";

    // Process
    unsigned char* output = new unsigned char[width * height * channels];
    auto start = std::chrono::high_resolution_clock::now();
    
    sobel_edge_detection(input, output, width, height, channels);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Processing time: " << elapsed.count() << "s\n";

    // Save with verification
    std::filesystem::path out_path(argv[2]);
    std::cout << "Saving to: " << std::filesystem::absolute(out_path) << std::endl;
    
    if (!stbi_write_png(argv[2], width, height, channels, output, width*channels)) {
        std::cerr << "Failed to write output (Error: " << stbi_failure_reason() << ")\n";
        
        // Fallback save
        std::string temp_path = "C:/temp/sobel_output.png";
        std::cout << "Trying fallback location: " << temp_path << std::endl;
        if (!stbi_write_png(temp_path.c_str(), width, height, channels, output, width*channels)) {
            std::cerr << "Critical error: Could not save anywhere!\n";
        }
    }

    // Cleanup
    stbi_image_free(input);
    delete[] output;
    return 0;
}
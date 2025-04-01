#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void sobel_edge_detection(const unsigned char* input, unsigned char* output, int width, int height, int channels) {
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            for (int c = 0; c < channels; c++) {
                // Calculate indices
                const int stride = width * channels;
                int i00 = (y-1)*stride + (x-1)*channels + c;
                int i01 = (y-1)*stride + x*channels + c;
                int i02 = (y-1)*stride + (x+1)*channels + c;
                int i10 = y*stride + (x-1)*channels + c;
                int i12 = y*stride + (x+1)*channels + c;
                int i20 = (y+1)*stride + (x-1)*channels + c;
                int i21 = (y+1)*stride + x*channels + c;
                int i22 = (y+1)*stride + (x+1)*channels + c;

                // Sobel X kernel
                float gx = -1.0f * input[i00] + 1.0f * input[i02] +
                           -2.0f * input[i10] + 2.0f * input[i12] +
                           -1.0f * input[i20] + 1.0f * input[i22];

                // Sobel Y kernel
                float gy = -1.0f * input[i00] - 2.0f * input[i01] - 1.0f * input[i02] +
                           1.0f * input[i20] + 2.0f * input[i21] + 1.0f * input[i22];

                // Gradient magnitude
                float mag = std::sqrt(gx*gx + gy*gy);
                output[y*stride + x*channels + c] = static_cast<unsigned char>(std::min(255.0f, 255.0f - mag));
            }
        }
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
    memset(output, 0, width * height * channels); // Initialize with black

    auto start = std::chrono::high_resolution_clock::now();
    sobel_edge_detection(input, output, width, height, channels);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Processing time: " << elapsed.count() << " seconds\n";

    // Save
    if (!stbi_write_png(argv[2], width, height, channels, output, width*channels)) {
        std::cerr << "Failed to write output\n";
    }

    // Cleanup
    stbi_image_free(input);
    delete[] output;
    return 0;
}
#include <stdint.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>

/**
 * @class BMP
 * @brief A class for creating and managing BMP image files.
 * 
 * This class provides functionality to create a BMP file, write pixel data,
 * and save the file to disk. It supports 24-bit and 32-bit BMP formats.
 * 
 * @authors Henrique Gabriel Rodrigues, Prof. Dr. André Rauber Du Bois
 */
class BMP
{
private:
    const int BITS_PER_PIXEL;
    const int BYTES_PER_PIXEL;

    std::ofstream file;
    const char *filename;
    uint8_t *pixelData;
    int height;
    int width;
    int rowSizeBytes; // Size of each row in bytes (including padding)

#pragma pack(1)
    struct HEADER
    {
        uint16_t bfType;      // Magic number for BMP files
        uint32_t bfSize;      // Size of the BMP file in bytes
        uint16_t bfReserved1; // Reserved; must be 0
        uint16_t bfReserved2; // Reserved; must be 0
        uint32_t bfOffBits;   // Offset to start of pixel data
    } header;

#pragma pack(1)
    struct INFOHEADER
    {
        uint32_t biSize;         // Size of this header (40 bytes)
        int32_t biWidth;         // Width of the bitmap in pixels
        int32_t biHeight;        // Height of the bitmap in pixels
        uint16_t biPlanes;       // Number of color planes (must be 1)
        uint16_t biBitCount;     // Bits per pixel (1, 4, 8, or 24)
        uint32_t biCompression;  // Compression type (0 = none)
        uint32_t biSizeImage;    // Size of the image data in bytes
        int32_t biXPelsPerMeter; // Horizontal resolution (pixels per meter)
        int32_t biYPelsPerMeter; // Vertical resolution (pixels per meter)
        uint32_t biClrUsed;      // Number of colors in the color palette
        uint32_t biClrImportant; // Important colors (0 = all)
    } infoHeader;

public:
    BMP(const char *filename, int width, int height, int bitsPerPixel = 24);
    ~BMP();

    void writePixel(int x, int y, uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255);
    void setPixelData(uint8_t *data);
    void save();
};
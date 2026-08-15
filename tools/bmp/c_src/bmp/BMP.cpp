#include "BMP.hpp"

BMP::BMP(const char *filename, int width, int height, int bitsPerPixel)
    : BITS_PER_PIXEL(bitsPerPixel), BYTES_PER_PIXEL(bitsPerPixel / 8)
{
    this->filename = filename;

    this->file.open(filename, std::ios::binary | std::ios::out);
    if (!this->file)
    {
        throw std::runtime_error("[BMP] Could not open file for writing");
    }

    this->width = width;
    this->height = height;
    this->rowSizeBytes = ((width * BYTES_PER_PIXEL + 3) & ~3); // Row size must be a multiple of 4 bytes

    // We are storing each color of the pixels as an 8-bit unsigned integer (0-255)
    this->pixelData = new uint8_t[rowSizeBytes * height];
    if (!this->pixelData)
    {
        throw std::runtime_error("[BMP] Could not allocate memory for pixel data");
    }

    // Set up the BMP file header
    this->header.bfType = 0x4D42; // 'BM'
    this->header.bfSize = sizeof(HEADER) + sizeof(INFOHEADER) + (rowSizeBytes * height);
    this->header.bfReserved1 = 0;
    this->header.bfReserved2 = 0;
    this->header.bfOffBits = sizeof(HEADER) + sizeof(INFOHEADER);

    // Set up the BMP info header
    this->infoHeader.biSize = sizeof(INFOHEADER);
    this->infoHeader.biWidth = width;
    this->infoHeader.biHeight = height;
    this->infoHeader.biPlanes = 1;
    this->infoHeader.biBitCount = this->BITS_PER_PIXEL; // Bits per pixel
    this->infoHeader.biCompression = 0;                 // No compression
    this->infoHeader.biSizeImage = 0;                   // Can be zero for uncompressed images
    this->infoHeader.biXPelsPerMeter = 2835;            // 72 DPI in PPM
    this->infoHeader.biYPelsPerMeter = 2835;            // 72 DPI in PPM
    this->infoHeader.biClrUsed = 0;                     // All colors are important
    this->infoHeader.biClrImportant = 0;                // All colors are important
}

BMP::~BMP()
{
    delete[] pixelData;
    if (file.is_open())
    {
        file.close();
    }
}

void BMP::writePixel(int x, int y, uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
    if (x < 0 || x >= width || y < 0 || y >= height)
    {
        throw std::out_of_range("[BMP] Pixel coordinates out of bounds");
    }

    int index = ((height - 1 - y) * width + x) * BYTES_PER_PIXEL; // BMP stores pixels bottom-up

    // Storing in BGR format
    pixelData[index] = b;     // Blue
    pixelData[index + 1] = g; // Green
    pixelData[index + 2] = r; // Red

    if (BYTES_PER_PIXEL == 4)
    {
        pixelData[index + 3] = a; // Alpha
    }
}

void BMP::setPixelData(uint8_t *data)
{
    std::memcpy(this->pixelData, data, this->rowSizeBytes * this->height);
}

void BMP::save()
{
    // Write the headers to the file
    this->file.write((char *)&this->header, sizeof(HEADER));
    this->file.write((char *)&this->infoHeader, sizeof(INFOHEADER));

    this->file.seekp(this->header.bfOffBits, std::ios::beg); // Move to the pixel data offset

    for (int y = 0; y < height; y++)
    {
        this->file.write((char *)&this->pixelData[y * this->rowSizeBytes], this->rowSizeBytes);
    }
}
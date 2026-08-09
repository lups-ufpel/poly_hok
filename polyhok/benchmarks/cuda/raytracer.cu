#include <stdio.h>
#include <time.h>
#include <stdint.h>
#define rnd(x) (x * rand() / RAND_MAX)
#define INF 2e10f
// asdf
#define _bitsperpixel 32
#define _planes 1
#define _compression 0

#define _xpixelpermeter 0x13B // 0x130B //2835 , 72 DPI
#define _ypixelpermeter 0x13B // 0x130B //2835 , 72 DPI
#define pixel 0xFF
#pragma pack(push, 1)
typedef struct
{
    uint8_t signature[2];
    uint32_t filesize;
    uint32_t reserved;
    uint32_t fileoffset_to_pixelarray;
} fileheader;
typedef struct
{
    uint32_t dibheadersize;
    uint32_t width;
    uint32_t height;
    uint16_t planes;
    uint16_t bitsperpixel;
    uint32_t compression;
    uint32_t imagesize;
    uint32_t ypixelpermeter;
    uint32_t xpixelpermeter;
    uint32_t numcolorspallette;
    uint32_t mostimpcolor;
} bitmapinfoheader;
typedef struct
{
    fileheader fileheader;
    bitmapinfoheader bitmapinfoheader;
} bitmap;
#pragma pack(pop)
void genBpm(int height, int width, int *pixelbuffer_f)
{
    uint32_t pixelbytesize = height * width * _bitsperpixel / 8;
    uint32_t _filesize = pixelbytesize + sizeof(bitmap);
    FILE *fp = fopen("raycuda.bmp", "wb");
    bitmap *pbitmap = (bitmap *)calloc(1, sizeof(bitmap));

    int buffer_size = height * width * 4;
    uint8_t *pixelbuffer = (uint8_t *)malloc(buffer_size);

    for (int i = 0; i < buffer_size; i++)
    {
        pixelbuffer[i] = (uint8_t)pixelbuffer_f[i];
    }

    // strcpy(pbitmap->fileheader.signature,"BM");
    pbitmap->fileheader.signature[0] = 'B';
    pbitmap->fileheader.signature[1] = 'M';
    pbitmap->fileheader.filesize = _filesize;
    pbitmap->fileheader.fileoffset_to_pixelarray = sizeof(bitmap);
    pbitmap->bitmapinfoheader.dibheadersize = sizeof(bitmapinfoheader);
    pbitmap->bitmapinfoheader.width = width;
    pbitmap->bitmapinfoheader.height = height;
    pbitmap->bitmapinfoheader.planes = _planes;
    pbitmap->bitmapinfoheader.bitsperpixel = _bitsperpixel;
    pbitmap->bitmapinfoheader.compression = _compression;
    pbitmap->bitmapinfoheader.imagesize = pixelbytesize;
    pbitmap->bitmapinfoheader.ypixelpermeter = _ypixelpermeter;
    pbitmap->bitmapinfoheader.xpixelpermeter = _xpixelpermeter;
    pbitmap->bitmapinfoheader.numcolorspallette = 0;
    fwrite(pbitmap, 1, sizeof(bitmap), fp);
    // memset(pixelbuffer,pixel,pixelbytesize);
    fwrite(pixelbuffer, 1, pixelbytesize, fp);
    fclose(fp);
    free(pbitmap);
    free(pixelbuffer);
}

struct Sphere
{
    float r, b, g;
    float radius;
    float x, y, z;
};

void loadSpheres(Sphere *vet, int size, float dim, float radius, float sum)
{

    for (int i = 0; i < size; i++)
    {
        Sphere sphere;
        sphere.r = rnd(1.0);
        sphere.b = rnd(1.0);
        sphere.g = rnd(1.0);
        sphere.radius = rnd(radius) + sum;
        sphere.x = rnd(dim) - trunc(dim / 2);
        sphere.y = rnd(dim) - trunc(dim / 2);
        sphere.z = rnd(256) - 128;

        vet[i] = sphere;
    }
}

#define SPHERES 20

__device__ void raytracing(int *image, int width, float *spheres, int x, int y)
{
    float ox = 0.0;
    float oy = 0.0;
    ox = (x - (width / 2));
    oy = (y - (width / 2));
    float r = 0.0;
    float g = 0.0;
    float b = 0.0;
    float maxz = (-99999.0);
    for (int i = 0; i < 20; i++)
    {
        float sphereRadius = spheres[((i * 7) + 3)];
        float dx = (ox - spheres[((i * 7) + 4)]);
        float dy = (oy - spheres[((i * 7) + 5)]);
        float n = 0.0;
        float t = (-99999.0);
        float dz = 0.0;
        if ((((dx * dx) + (dy * dy)) < (sphereRadius * sphereRadius)))
        {
            dz = sqrtf((((sphereRadius * sphereRadius) - (dx * dx)) - (dy * dy)));
            n = (dz / sqrtf((sphereRadius * sphereRadius)));
            t = (dz + spheres[((i * 7) + 6)]);
        }
        else
        {
            t = (-99999.0);
            n = 0.0;
        }

        if ((t > maxz))
        {
            float fscale = n;
            r = (spheres[((i * 7) + 0)] * fscale);
            g = (spheres[((i * 7) + 1)] * fscale);
            b = (spheres[((i * 7) + 2)] * fscale);
            maxz = t;
        }
    }

    image[0] = (r * 255);
    image[1] = (g * 255);
    image[2] = (b * 255);
    image[3] = 255;
}

extern "C" __global__ void mapxy_2D_step_2_para_no_resp_kernel(int *d_array, int step, int par1, float *par2, int size)
{
    int x = (threadIdx.x + (blockIdx.x * blockDim.x));
    int y = (threadIdx.y + (blockIdx.y * blockDim.y));
    int offset = (x + ((y * blockDim.x) * gridDim.x));
    int id = (step * offset);
    if ((offset < (size * size)))
    {
        raytracing((d_array + id), par1, par2, x, y);
    }
}

int main(int argc, char *argv[])
{
    int dim = atoi(argv[1]);
    // int sph = atoi(argv[2]);

    int *final_image;
    int *dev_image;
    Sphere *s;

    final_image = (int *)malloc((size_t)dim * dim * sizeof(int) * 4);
    Sphere *temp_s = (Sphere *)malloc(sizeof(Sphere) * SPHERES);

    loadSpheres(temp_s, SPHERES, dim, 160, 20);

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMalloc((void **)&dev_image, (size_t)dim * dim * sizeof(int) * 4);
    cudaMalloc((void **)&s, sizeof(Sphere) * SPHERES);

    cudaMemcpy(s, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice);

    //  dim3    grids(dim/16,dim/16);
    //  dim3    threads(16,16);

    //  mapxy_2D_step_2_para_no_resp_kernel<<<grids,threads>>>(dev_image, 4,dim,((float*) s), dim);

    dim3 grids(dim, dim);
    //   dim3    threads(16,16);

    mapxy_2D_step_2_para_no_resp_kernel<<<grids, 1>>>(dev_image, 4, dim, ((float *)s), dim);

    // kernel<<<grids,threads>>>(dim, s, dev_image);

    cudaMemcpy(final_image, dev_image, (size_t)dim * dim * sizeof(int) * 4, cudaMemcpyDeviceToHost);

    cudaFree(dev_image);
    cudaFree(s);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("CUDA\t%d\t%3.1f\n", dim, time);
    // genBpm(dim,dim,final_image);

    free(temp_s);
    free(final_image);
}
/*
    This NIF module provides functions to generate BMP image files from pixel data.

    Made by: Henrique Gabriel Rodrigues
    Oriented and supervised by: Prof. Dr. André Rauber Du Bois
*/

#include <erl_nif.h>
#include <cmath>

#include "bmp/BMP.hpp"

void genBpm(int height, int width, uint8_t *pixelbuffer, const char *file_name)
{
  BMP bmp(file_name, width, height, 32);
  bmp.setPixelData(pixelbuffer);
  bmp.save();
}

static ERL_NIF_TERM gen_bmp_int_nif(ErlNifEnv *env, int /* argc */, const ERL_NIF_TERM argv[])
{
  ErlNifBinary matrix_el;
  int dim;

  /// GET FILE NAME
  ERL_NIF_TERM list = argv[0];

  unsigned int size;
  enif_get_list_length(env, list, &size);

  char file_name[1024];
  enif_get_string(env, list, file_name, size + 1, ERL_NIF_LATIN1);

  /// GET DIM
  if (!enif_get_int(env, argv[1], &dim))
  {
    return enif_make_badarg(env);
  }

  /// GET PIXEL MATRIX
  if (!enif_inspect_binary(env, argv[2], &matrix_el))
  {
    return enif_make_badarg(env);
  }

  // CREATING THE PIXEL BUFFER
  size_t matrex_size = dim * dim * 4;
  uint8_t *pixelbuffer = new uint8_t[matrex_size];
  int *matrix = (int *)matrix_el.data;

  for (size_t i = 0; i < matrex_size; i++)
  {
    pixelbuffer[i] = static_cast<uint8_t>(matrix[i]);
  }

  genBpm(dim, dim, pixelbuffer, file_name);

  delete[] pixelbuffer;
  return enif_make_int(env, 0);
}

static ERL_NIF_TERM gen_bmp_float_nif(ErlNifEnv *env, int /* argc */, const ERL_NIF_TERM argv[])
{
  ErlNifBinary matrix_el;
  int dim;

  /// GET FILE NAME
  ERL_NIF_TERM list = argv[0];

  unsigned int size;
  enif_get_list_length(env, list, &size);
  char file_name[1024];

  enif_get_string(env, list, file_name, size + 1, ERL_NIF_LATIN1);

  /// GET DIM
  if (!enif_get_int(env, argv[1], &dim))
  {
    return enif_make_badarg(env);
  }

  /// GET PIXEL MATRIX
  if (!enif_inspect_binary(env, argv[2], &matrix_el))
  {
    return enif_make_badarg(env);
  }

  // CREATING THE PIXEL BUFFER
  size_t matrex_size = dim * dim * 4;
  uint8_t *pixelbuffer = new uint8_t[matrex_size];

  float *matrix = (float *)matrix_el.data;
  matrix += 2;

  for (size_t i = 0; i < matrex_size; i++)
  {
    pixelbuffer[i] = static_cast<uint8_t>(matrix[i]);
  }

  genBpm(dim, dim, pixelbuffer, file_name);

  delete[] pixelbuffer;
  return enif_make_int(env, 0);
}

// Creates a test BMP image just to demonstrate the functionality of the NIF module. The size will be dim x dim
// Param 1: Filename (as charlist)
// Param 2: Dimension of the image (int)
static ERL_NIF_TERM test_bmp_generation_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  if (argc != 2)
  {
    return enif_make_badarg(env);
  }

  // Get filename
  ERL_NIF_TERM e_filename = argv[0];
  unsigned int size;

  if (!enif_get_list_length(env, e_filename, &size))
  {
    return enif_make_badarg(env);
  }

  std::string filename(size, '\0');
  enif_get_string(env, e_filename, filename.data(), size + 1, ERL_NIF_LATIN1);

  // Get dimension
  uint dim;
  if (!enif_get_uint(env, argv[1], &dim))
  {
    return enif_make_badarg(env);
  }

  // Creating a pixel buffer with size dim x dim, where each pixel has 4 channels (RGBA)
  // Therefore, dim x dim x 4 elements
  uint8_t *pixelbuffer = new uint8_t[dim * dim * 4];

  for (int i = 0; i < dim; i++)
  {
    for (int j = 0; j < dim; j++)
    {
      uint8_t b = static_cast<uint8_t>((int)(i * 0.5) % 256);
      uint8_t g = static_cast<uint8_t>((int)(j * 0.5) % 256);
      uint8_t r = static_cast<uint8_t>((int)(i + j) % 256);
      uint8_t a = static_cast<uint8_t>(255);

      pixelbuffer[(i * dim + j) * 4 + 0] = b; // B
      pixelbuffer[(i * dim + j) * 4 + 2] = g; // G
      pixelbuffer[(i * dim + j) * 4 + 1] = r; // R
      pixelbuffer[(i * dim + j) * 4 + 3] = a; // A
    }
  }

  genBpm(dim, dim, pixelbuffer, filename.c_str());

  delete[] pixelbuffer;
  return enif_make_int(env, 0);
}

static ErlNifFunc nif_funcs[] = {
    {"gen_bmp_int_nif", 3, gen_bmp_int_nif, 0},
    {"gen_bmp_float_nif", 3, gen_bmp_float_nif, 0},
    {"test_bmp_generation_nif", 2, test_bmp_generation_nif, 0}};

ERL_NIF_INIT(Elixir.Bmp, nif_funcs, NULL, NULL, NULL, NULL)
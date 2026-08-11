/*
    This NIF module provides functions to generate BMP image files from pixel data.

    Made by: Henrique Gabriel Rodrigues
    Oriented and supervised by: Prof. Dr. André Rauber Du Bois
*/

#include <erl_nif.h>

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

static ErlNifFunc nif_funcs[] = {
    {.name = "gen_bmp_int_nif", .arity = 3, .fptr = gen_bmp_int_nif, .flags = 0},
    {.name = "gen_bmp_float_nif", .arity = 3, .fptr = gen_bmp_float_nif, .flags = 0}};

ERL_NIF_INIT(Elixir.BMP, nif_funcs, NULL, NULL, NULL, NULL)
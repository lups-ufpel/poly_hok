defmodule PolyHok.BackendBehaviour do
  @type body_ast :: tuple()
  @type code_body :: String.t()
  @type types :: map()
  @type param_vars :: list()
  @type module_name :: String.t()
  @type subs :: map()
  @type name :: String.t()
  @type param_list_str :: String.t()
  @type return_type :: atom()

  # Previously called 'gen_cuda_jit' and 'gen_ocl_jit'
  @callback gen_code(body_ast, types, param_vars, module_name, subs) :: String.t()

  # Previously called "gen_function"
  @callback declare_function(name, param_list_str, code_body, return_type) :: String.t()

  # Previously called "gen_kernel"
  @callback declare_kernel(name, param_list_str, code_body) :: String.t()

  # Correctly declares the parameters variables based on types in the target language
  @type p :: atom()
  @type fun_type :: tuple()

  @callback gen_para(p, :tint) :: String.t()
  @callback gen_para(p, :tfloat) :: String.t()
  @callback gen_para(p, :tdouble) :: String.t()
  @callback gen_para(p, :int) :: String.t()
  @callback gen_para(p, :float) :: String.t()
  @callback gen_para(p, :double) :: String.t()
  @callback gen_para(p, fun_type) :: String.t() | nil

  # -------------- NIFs ----------------
  @type l :: integer()
  @type c :: integer()
  @type type :: charlist()

  @type kernel_name :: charlist()
  @type kernel_code :: charlist()
  @type kernel_resource :: any()

  @type blocks :: tuple()
  @type threads :: tuple()
  @type len_args :: integer()
  @type types_args :: list()
  @type args :: list()

  @type nx :: any()
  @type gnx :: any()

  @callback new_empty_gpu_array_nif(l, c, type) :: any()
  @callback new_gpu_array_from_nx_nif(nx, l, c, type) :: any()
  @callback get_gpu_array_nif(gnx, l, c, type) :: any()
  @callback synchronize_nif() :: any()
  @callback jit_compile_nif(kernel_name, kernel_code) :: any()
  @callback jit_launch_nif(kernel_resource, blocks, threads, len_args, types_args, args) :: any()

  # -------------- Optional Callbacks ----------------
  @callback set_debug_logs_nif(boolean()) :: integer()
  @callback double_supported_nif() :: boolean()

  @optional_callbacks [set_debug_logs_nif: 1, double_supported_nif: 0]
end

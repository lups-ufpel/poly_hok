defmodule PolyHok.BackendBehaviour do
  @type header :: tuple()
  @type body :: tuple()
  @type types :: map()
  @type param_vars :: list()
  @type module_name :: String.t()
  @type subs :: map()
  @type name :: String.t()
  @type param_list_str :: String.t()
  @type return_type :: atom()

  @callback gen_new_module(header, body) :: tuple()

  # Previously called 'gen_cuda_jit' and 'gen_ocl_jit'
  @callback gen_code(body, types, param_vars, module_name, subs) :: String.t()

  # Previously called "gen_function"
  @callback declare_function(name, param_list_str, body, return_type) :: String.t()

  # Previously called "gen_kernel"
  @callback declare_kernel(name, param_list_str, body) :: String.t()

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
end

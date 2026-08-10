defmodule PolyHok.BackendBehaviour do
  @callback gen_new_module(header, body) :: tuple()

  # Previously called 'gen_cuda_jit' and 'gen_ocl_jit'
  @callback gen_code(body, types, param_vars, module, subs) :: String.t()

  # Previously called "gen_function"
  @callback declare_function(name, para, body, return_type) :: String.t()

  # Previously called "gen_kernel"
  @callback declare_kernel(name, para, body) :: String.t()

  # Correctly declares the parameters variables based on types in the target language
  @callback gen_para(p, :tint)
  @callback gen_para(p, :tfloat)
  @callback gen_para(p, :tdouble)
  @callback gen_para(p, :int)
  @callback gen_para(p, :float)
  @callback gen_para(p, :double)
end

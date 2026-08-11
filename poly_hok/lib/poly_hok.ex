defmodule PolyHok do
  @on_load :load_nifs
  def load_nifs do
    :erlang.load_nif("./priv/gpu_nifs", 0)
  end

  defmacro clo({:fn, aa, [{:->, bb, [para, body]}]}) do
    # IO.inspect "body: #{inspect body}"
    # raise "hell"
    body = PolyHok.CudaBackend.add_return(body)
    funs = JIT.find_functions({:fn, aa, [{:->, bb, [para, body]}]})
    name = "anon_" <> PolyHok.CudaBackend.gen_lambda_name()

    free = JIT.find_free_vars({:fn, aa, [{:->, bb, [para, body]}]})

    extra =
      free
      |> Enum.map(fn p -> {p, [], nil} end)

    free = Enum.map(free, fn p -> String.to_atom(name <> Atom.to_string(p)) end)
    function = {:fn, aa, [{:->, bb, [para ++ extra, body]}]}

    # IO.inspect list
    resp =
      quote(
        do:
          {:closure, unquote(name), {unquote(Macro.escape(function)), unquote(funs)},
           unquote(free), unquote(extra)}
      )

    #  resp =  quote(do: {:anon , unquote(name),unquote({:fn, aa, [{:->, bb , [para,body]}] })})
    resp
  end

  defmacro phok({:fn, aa, [{:->, bb, [para, body]}]}) do
    # IO.inspect "body: #{inspect body}"
    body = PolyHok.CudaBackend.add_return(body)
    funs = JIT.find_functions({:fn, aa, [{:->, bb, [para, body]}]})
    name = "anon_" <> PolyHok.CudaBackend.gen_lambda_name()
    function = {:fn, aa, [{:->, bb, [para, body]}]}
    resp = quote(do: {:anon, unquote(name), {unquote(Macro.escape(function)), unquote(funs)}})
    #  resp =  quote(do: {:anon , unquote(name),unquote({:fn, aa, [{:->, bb , [para,body]}] })})
    resp
    # IO.inspect function
    # raise "hell"
    # {fname,type} = PolyHok.CudaBackend.gen_lambda("Elixir.App",function)
    # result = quote do: PolyHok.load_lambda_compilation(unquote("Elixir.App"), unquote(fname), unquote(type))
    # result
    # IO.inspect result
    # raise "hell"
  end

  defmacro gpu_for({:<-, _, [var, tensor]}, do: b) do
    quote do:
            PolyHok.new_gnx(unquote(tensor))
            |> PMap.map(PolyHok.phok(fn unquote(var) -> unquote(b) end))
            |> PolyHok.get_gnx()
  end

  defmacro gpu_for({:<-, _, [var1, {:.., _, [_b1, e1]}]}, arr1, arr2, do: body) do
    r =
      quote do:
              PMap.comp_func(
                unquote(arr1),
                unquote(arr2),
                unquote(e1),
                PolyHok.phok(fn unquote(arr1), unquote(arr2), unquote(var1) -> unquote(body) end)
              )

    # IO.inspect r
    # raise "hell"
    r
  end

  # defmacro gpu_for({:<-, _ ,[var,tensor]},do: b)  do
  #    quote do: PolyHok.new_gnx(unquote(tensor))
  #              |> PMap.map(PolyHok.phok (fn (unquote(var)) -> (unquote b) end))
  #              |> PolyHok.get_gnx
  # end
  defmacro gpufor({:<-, _, [var, tensor]}, do: b) do
    quote do: Comp.comp(unquote(tensor), PolyHok.phok(fn unquote(var) -> unquote(b) end))
  end

  defmacro gpufor({:<-, _, [var1, {:.., _, [_b1, e1]}]}, arr1, arr2, do: body) do
    r =
      quote do:
              Comp.comp_xy_2arrays(
                unquote(arr1),
                unquote(arr2),
                unquote(e1),
                PolyHok.phok(fn unquote(arr1), unquote(arr2), unquote(var1) -> unquote(body) end)
              )

    # IO.inspect r
    # raise "hell"
    r
  end

  defmacro gpufor(
             {:<-, _, [var1, {:.., _, [_b1, e1]}]},
             {:<-, _, [var2, {:.., _, [_b2, e2]}]},
             arr1,
             arr2,
             par3,
             do: body
           ) do
    # IO.inspect "Aqui"
    r =
      quote do:
              MM.comp2xy2D1p(
                unquote(arr1),
                unquote(arr2),
                unquote(par3),
                unquote(e1),
                unquote(e2),
                PolyHok.phok(fn unquote(arr1),
                                unquote(arr2),
                                unquote(par3),
                                unquote(var1),
                                unquote(var2) ->
                  unquote(body)
                end)
              )

    # IO.inspect r
    # raise "hell"
    r
  end

  defmacro defmodule(header, do: body) do
    {:__aliases__, _, [module_name]} = header
    JIT.process_module(module_name, body)

    ast_new_module = PolyHok.CudaBackend.gen_new_module(header, body)
    ast_new_module
  end

  def get_type_gnx({:nx, type, _shape, _name, _ref}) do
    type
  end

  def get_type({:nx, type, _shape, _name, _ref}) do
    type
  end

  def get_shape_gnx({:nx, _type, shape, _name, _ref}) do
    shape
  end

  def get_shape({:nx, _type, shape, _name, _ref}) do
    shape
  end

  def new_gnx(%Nx.Tensor{data: data, type: type, shape: shape, names: name}) do
    %Nx.BinaryBackend{state: array} = data
    # IO.inspect name
    # raise "hell"
    {l, c} =
      case shape do
        {c} -> {1, c}
        {l, c} -> {l, c}
        {l1, l2, c} -> {l1 * l2, c}
      end

    ref =
      case type do
        {:f, 32} -> create_gpu_array_nx_nif(array, l, c, Kernel.to_charlist("float"))
        {:f, 64} -> create_gpu_array_nx_nif(array, l, c, Kernel.to_charlist("double"))
        {:s, 32} -> create_gpu_array_nx_nif(array, l, c, Kernel.to_charlist("int"))
        x -> raise "new_gnx: type #{x} not suported"
      end

    {:nx, type, shape, name, ref}
  end

  def new_gnx(l, c, type) do
    # IO.puts "aque"
    ref =
      case type do
        {:f, 32} -> new_gpu_array_nif(l, c, Kernel.to_charlist("float"))
        {:f, 64} -> new_gpu_array_nif(l, c, Kernel.to_charlist("double"))
        {:s, 32} -> new_gpu_array_nif(l, c, Kernel.to_charlist("int"))
        x -> raise "new_gnx: type #{x} not suported"
      end

    {:nx, type, {l, c}, [nil, nil], ref}
  end

  def new_gnx({c}, type) do
    l = 1
    # IO.puts "aque"
    ref =
      case type do
        {:f, 32} -> new_gpu_array_nif(l, c, Kernel.to_charlist("float"))
        {:f, 64} -> new_gpu_array_nif(l, c, Kernel.to_charlist("double"))
        {:s, 32} -> new_gpu_array_nif(l, c, Kernel.to_charlist("int"))
        x -> raise "new_gnx: type #{x} not suported"
      end

    {:nx, type, {c}, [nil], ref}
  end

  def new_gnx({l, c}, type) do
    # IO.puts "aque"
    ref =
      case type do
        {:f, 32} -> new_gpu_array_nif(l, c, Kernel.to_charlist("float"))
        {:f, 64} -> new_gpu_array_nif(l, c, Kernel.to_charlist("double"))
        {:s, 32} -> new_gpu_array_nif(l, c, Kernel.to_charlist("int"))
        x -> raise "new_gnx: type #{x} not suported"
      end

    {:nx, type, {l, c}, [nil, nil], ref}
  end

  def new_gnx({d1, d2, d3}, type) do
    {l, c} = {d1 * d2, d3}

    ref =
      case type do
        {:f, 32} -> new_gpu_array_nif(l, c, Kernel.to_charlist("float"))
        {:f, 64} -> new_gpu_array_nif(l, c, Kernel.to_charlist("double"))
        {:s, 32} -> new_gpu_array_nif(l, c, Kernel.to_charlist("int"))
        x -> raise "new_gnx: type #{x} not suported"
      end

    {:nx, type, {d1, d2, d3}, [nil, nil, nil], ref}
  end

  def get_gnx({:nx, type, shape, name, ref}) do
    # IO.puts "aqui..."
    {l, c} =
      case shape do
        {c} -> {1, c}
        {l, c} -> {l, c}
        {d1, d2, d3} -> {d1 * d2, d3}
      end

    ref =
      case type do
        {:f, 32} -> get_gpu_array_nif(ref, l, c, Kernel.to_charlist("float"))
        {:f, 64} -> get_gpu_array_nif(ref, l, c, Kernel.to_charlist("double"))
        {:s, 32} -> get_gpu_array_nif(ref, l, c, Kernel.to_charlist("int"))
        x -> raise "new_gnx: type #{x} not suported"
      end

    %Nx.Tensor{data: %Nx.BinaryBackend{state: ref}, type: type, shape: shape, names: name}
  end

  def new_nx_from_function(l, c, type, fun) do
    size = l * c

    ref =
      case type do
        {:f, 32} -> new_matrix_from_function_f(size - 1, fun, <<fun.()::float-little-32>>)
        {:f, 64} -> new_matrix_from_function_d(size - 1, fun, <<fun.()::float-little-64>>)
        {:s, 32} -> new_matrix_from_function_i(size - 1, fun, <<fun.()::integer-little-32>>)
      end

    %Nx.Tensor{data: %Nx.BinaryBackend{state: ref}, type: type, shape: {l, c}, names: [nil, nil]}
  end

  #######################
  defp new_matrix_from_function_d(0, _, accumulator), do: accumulator

  defp new_matrix_from_function_d(size, function, accumulator),
    do:
      new_matrix_from_function_d(
        size - 1,
        function,
        <<accumulator::binary, function.()::float-little-64>>
      )

  defp new_matrix_from_function_i(0, _, accumulator), do: accumulator

  defp new_matrix_from_function_i(size, function, accumulator),
    do:
      new_matrix_from_function_i(
        size - 1,
        function,
        <<accumulator::binary, function.()::integer-little-32>>
      )

  defp new_matrix_from_function_f(0, _, accumulator), do: accumulator

  defp new_matrix_from_function_f(size, function, accumulator),
    do:
      new_matrix_from_function_f(
        size - 1,
        function,
        <<accumulator::binary, function.()::float-little-32>>
      )

  ##############################
  def new_nx_from_function_arg(l, c, type, fun) do
    size = l * c

    ref =
      case type do
        {:f, 32} ->
          new_matrix_from_function_f_arg(size - 1, fun, <<fun.(size)::float-little-32>>)

        {:f, 64} ->
          new_matrix_from_function_d_arg(size - 1, fun, <<fun.(size)::float-little-64>>)

        {:s, 32} ->
          new_matrix_from_function_i_arg(size - 1, fun, <<fun.(size)::integer-little-32>>)
      end

    %Nx.Tensor{data: %Nx.BinaryBackend{state: ref}, type: type, shape: {l, c}, names: [nil, nil]}
  end

  #######################
  defp new_matrix_from_function_d_arg(0, _, accumulator), do: accumulator

  defp new_matrix_from_function_d_arg(size, function, accumulator),
    do:
      new_matrix_from_function_d_arg(
        size - 1,
        function,
        <<accumulator::binary, function.(size)::float-little-64>>
      )

  defp new_matrix_from_function_i_arg(0, _, accumulator), do: accumulator

  defp new_matrix_from_function_i_arg(size, function, accumulator),
    do:
      new_matrix_from_function_i_arg(
        size - 1,
        function,
        <<accumulator::binary, function.(size)::integer-little-32>>
      )

  defp new_matrix_from_function_f_arg(0, _, accumulator), do: accumulator

  defp new_matrix_from_function_f_arg(size, function, accumulator),
    do:
      new_matrix_from_function_f_arg(
        size - 1,
        function,
        <<accumulator::binary, function.(size)::float-little-32>>
      )

  def synchronize() do
    synchronize_nif()
  end

  ############################################################## Loading types and asts from files
  def load_ast(kernel) do
    {_module, f_name} =
      case Macro.escape(kernel) do
        {:&, [], [{:/, [], [{{:., [], [module, f_name]}, [no_parens: true], []}, _nargs]}]} ->
          {module, f_name}

        f ->
          {:ok, f}
          # _ -> raise "Argument to spawn should be a function."
      end

    # bytes = File.read!("c_src/#{module}.asts")
    # map_asts = :erlang.binary_to_term(bytes)
    # IO.inspect map_size(map_asts)
    # {ast,_typed?,_types} = Map.get(map_asts,String.to_atom("#{f_name}"))
    # ast

    send(:module_server, {:get_ast, f_name, self()})

    receive do
      {:ast, ast} -> ast
      h -> raise "unknown message for function type server #{inspect(h)}"
    end
  end

  #########################
  defp process_args_no_fun([{:anon, _name, _type} | t1]) do
    process_args_no_fun(t1)
  end

  defp process_args_no_fun([{:nx, _type, _shape, _name, ref} | t1]) do
    [ref | process_args_no_fun(t1)]
  end

  defp process_args_no_fun([arg | t1]) when is_function(arg) do
    process_args_no_fun(t1)
  end

  defp process_args_no_fun([arg | t1]) do
    [arg | process_args_no_fun(t1)]
  end

  defp process_args_no_fun([]), do: []

  # ----------------- JIT compilation and kernel spawning -----------------

  @doc """
  Spwans a kernel with JIT compilation.

  Generates the OpenCL kernel code for the given kernel, compiles it, and queues it for execution.

  ## Parameters

    - `k`: The kernel function to be compiled and executed.
    - `t`: The work group size in each dimension (a.k.a number of blocks).
    - `b`: A list containing the number of work items in each dimension (a.k.a threads per block).
    - `l`: A list of arguments to be passed to the kernel.
  """
  def spawn(k, t, b, l) do
    # Get kernel name from the kernel function reference.
    kernel_name = JIT.get_kernel_name(k)

    # Load, from the module_server, the AST and function graph for the kernel.
    {kast, fun_graph} =
      case load_ast(k) do
        {a, g} -> {a, g}
        nil -> raise "Unknown kernel #{inspect(kernel_name)}"
      end

    # Eliminate clojures
    {kast, l} = JIT.closure_elimination(kast, l)

    # Generates initial delta based on the types of the actual parameters
    initial_delta = JIT.gen_types_delta(kast, l)
    # Map of kernel_function_para -> actual_name_in_code
    subs = JIT.get_function_parameters(kast, l)

    # ============ temp debug
    IO.inspect(initial_delta, label: "initial delta")
    IO.inspect(subs, label: "subs")

    System.halt()

    # kernel_types_funs = Map.merge(delta, subs) |> Map.to_list()
    # map_key = {kernel_name, kernel_types_funs}

    # send(:module_server, {:get_kernel, map_key, self()})

    # # ---------------------------------------------------

    # {kernel_res, types_args} =
    #   receive do
    #     {:kernel, nil} ->
    #       fun_graph_asts_sorted =
    #         JIT.get_non_parameters_func_asts(fun_graph)
    #         # Now we need to sort these functions in the correct order of inference
    #         |> JIT.sort_functions_by_call_graph()

    #       inner_funs_delta = JIT.infer_device_functions_types(fun_graph_asts_sorted)

    #       delta = Map.merge(initial_delta, inner_funs_delta)

    #       # Infers the types of the kernel's variables using the new delta map
    #       inf_types =
    #         case JIT.infer_types(kast, delta, kernel_name) do
    #           {:ok, types} -> types
    #           {:error, _types, reason} -> raise "Type inference failed: #{reason}"
    #         end

    #       ## --------------- THE INFAMOUS DOUBLE CHECK ----------------------##
    #       # I'll have to figure a way to do this across different backends. This
    #       # check is specifically for OCL-PolyHok...

    #       # # Check if the inferred types contain 'double' or 'tdouble' types
    #       # contains_double =
    #       #   Map.values(inf_types) |> Enum.any?(fn x -> x == :double or x == :tdouble end)

    #       # # If double precision is used, check if the device supports it.
    #       # if contains_double and not double_supported_nif() do
    #       #   raise "[OCL-PolyHok] Your OpenCL device does not support double precision floating point operations (fp64). The 'double' data type cannot be used in kernels."
    #       # end

    #       ## --------------------------------------------------------------- ##

    #       # Generates kernel string in CUDA/OpenCL/etc
    #       kernel = JIT.compile_kernel(kast, inf_types, subs)

    #       # Get a list of tuples {actual_function_param, type} for all formal parameters that are functions.
    #       param_funs = JIT.get_function_parameters_and_their_types(kast, l, inf_types)

    #       # Creates a list of tuples where each tuple contains a function name and its inferred type signature
    #       other_funs =
    #         fun_graph_asts_sorted
    #         |> Enum.map(fn {x, _ast} -> {x, inf_types[x]} end)
    #         # Remove functions that could not be inferred
    #         |> Enum.filter(fn {_, i} -> i != nil end)

    #       all_funs = other_funs ++ param_funs

    #       # The JIT.compile_function/2 function compiles the provided function AND it's dependencies (other functions called within
    #       # a function). To avoid recompiling functions that were already compiled, we provide a MapSet of already compiled functions,
    #       # so the JIT.compile_function/2 can check and skip a function if necessary.
    #       # We also re-infer the device functions here now that we have the kernel delta to guarantee we have the correct types
    #       {comp, _compiled_funs} =
    #         Enum.reduce(all_funs, {[], MapSet.new()}, fn fun, {code_acc, compiled_funs_acc} ->
    #           {new_code, compiled_funs_acc} = JIT.compile_function(fun, compiled_funs_acc)
    #           {code_acc ++ new_code, compiled_funs_acc}
    #         end)

    #       includes = JIT.get_includes()
    #       prog = [includes | comp] ++ [kernel]

    #       # Concatenating the generated code into a single string
    #       prog = Enum.reduce(prog, "", fn x, y -> y <> x end)

    #       # Print generated code for debugging purposes if debug logs are enabled
    #       debug_logs = Agent.get(:debug_logs_agent, fn state -> state end)

    #       if debug_logs do
    #         IO.puts("===== Generated code for kernel '#{kernel_name}' =====")

    #         # We don't print the includes to reduce clutter
    #         case comp do
    #           [] -> IO.puts(kernel)
    #           l -> IO.puts(Enum.reduce(l, "", fn x, y -> y <> x end) <> kernel)
    #         end

    #         IO.puts("==============================================================")
    #       end

    #       # List of the actual arguments passed to the kernel except functions
    #       args = process_args_no_fun(l)
    #       # List of the inferred types for 'args'
    #       types_args = JIT.get_types_para(kast, inf_types)
    #   end
  end

  # ------------------------------ NIF Stubs ------------------------------

  defp new_gpu_array_nif(_l, _c, _type) do
    :erlang.nif_error(:nif_not_loaded)
  end

  defp get_gpu_array_nif(_matrex, _l, _c, _type) do
    :erlang.nif_error(:nif_not_loaded)
  end

  defp create_gpu_array_nx_nif(_matrex, _l, _c, _type) do
    :erlang.nif_error(:nif_not_loaded)
  end

  defp synchronize_nif() do
    :erlang.nif_error(:nif_not_loaded)
  end

  def jit_compile_and_launch_nif(_n, _k, _t, _b, _size, _types, _l) do
    :erlang.nif_error(:nif_not_loaded)
  end
end

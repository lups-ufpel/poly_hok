defmodule PolyHok do
  @moduledoc """
  This is PolyHok's core module. It provides the main interface for the
  PolyHok DSL, containing macros for defining PolyHok modules, GPU kernels
  and device functions, as well as functions for creating and manipulating
  GNx (GPU Nx) tensors.

  This module does not contain any API-specific implementation (e.g
  OpenCL, CUDA, etc). Instead, we delegate this implementation to the
  backend module, that must implement the `PolyHok.BackendBehaviour`
  behaviour.

  The user can set the backend module it wants to use by configuring
  the config/runtime.exs file. We then load the backend module at
  runtime using `:persistent_term` storage and delegate the calls
  to it. This allows PolyHok to be decoupled from any specific
  backend implementation, and makes it easy to add new backends
  in the future.
  """

  @doc """
  Returns the current backend module used by PolyHok.

  Runs in constant O(1) time, as it uses persistent_term storage.
  """
  @spec backend() :: module()
  def backend(), do: :persistent_term.get({PolyHok, :backend})

  @doc """
  Check if a given function is implemented in the backend.

  Used for optional callbacks in the backend behaviour.
  """
  @spec exists_in_backend?(atom(), arity()) :: boolean()
  def exists_in_backend?(function_name, arity) do
    b = backend()
    Code.ensure_loaded?(b) and function_exported?(b, function_name, arity)
  end

  defmacro clo({:fn, aa, [{:->, bb, [para, body]}]}) do
    body = PolyHok.TypeInference.add_return(body)
    funs = JIT.find_functions({:fn, aa, [{:->, bb, [para, body]}]})
    name = "anon_" <> gen_lambda_name()

    free = JIT.find_free_vars({:fn, aa, [{:->, bb, [para, body]}]})

    extra =
      free
      |> Enum.map(fn p -> {p, [], nil} end)

    free = Enum.map(free, fn p -> String.to_atom(name <> Atom.to_string(p)) end)
    function = {:fn, aa, [{:->, bb, [para ++ extra, body]}]}

    resp =
      quote(
        do:
          {:closure, unquote(name), {unquote(Macro.escape(function)), unquote(funs)},
           unquote(free), unquote(extra)}
      )

    resp
  end

  defmacro phok({:fn, aa, [{:->, bb, [para, body]}]}) do
    # Adding return to body and update fun
    new_body = PolyHok.TypeInference.add_return(body)
    fun = {:fn, aa, [{:->, bb, [para, new_body]}]}

    # Find inner functions calls and create lambda name
    inner_funs = JIT.find_functions(fun)
    lambda_name = "anon_" <> gen_lambda_name()

    # Return formatted as {:anon, lambda_name, {fun, inner_funs}}
    quote do
      {:anon, unquote(lambda_name), {unquote(Macro.escape(fun)), unquote(inner_funs)}}
    end
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

    r
  end

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

    r
  end

  defmacro defmodule(header, do: body) do
    {:__aliases__, _, [module_name]} = header
    JIT.process_module(module_name, body)

    ast_new_module = backend().gen_new_module(header, body)
    ast_new_module
  end

  @doc """
  Creates a unique name for anonymous functions (lambdas) by generating a random string of 10 characters.

  ## Returns

    - A string of 10 random characters chosen from the set "0123456789abcdefghijklmno".
  """
  def gen_lambda_name() do
    for _ <- 1..10, into: "", do: <<Enum.random(~c"0123456789abcdefghijklmno")>>
  end

  # ----------------- Debug Logs -----------------

  @dialyzer {:nowarn_function, set_debug_logs: 1}
  def set_debug_logs(enable) do
    Agent.update(:debug_logs_agent, fn _old -> enable end)

    if exists_in_backend?(:set_debug_logs_nif, 1) do
      backend().set_debug_logs_nif(enable)
    end

    :ok
  end

  # ----------------- GPU NX miscellaneous functions -----------------

  def get_type_gnx({:nx, type, _shape, _name, _ref}), do: type

  def get_type(%Nx.Tensor{type: type}), do: type

  def get_shape_gnx({:nx, _type, shape, _name, _ref}), do: shape

  def get_shape(%Nx.Tensor{shape: shape}), do: shape

  # ------- Helper functions for GNx creation -------

  defp get_type_charlist(type) do
    case type do
      t when t in [{:f, 32}, :f32] -> Kernel.to_charlist("float")
      t when t in [{:f, 64}, :f64] -> Kernel.to_charlist("double")
      t when t in [{:s, 32}, :s32] -> Kernel.to_charlist("int")
      x -> raise "PolyHok: type #{inspect(x)} is not suported"
    end
  end

  defp get_lines_cols(shape) do
    case shape do
      {c} -> {1, c}
      {l, c} -> {l, c}
      {l1, l2, c} -> {l1 * l2, c}
    end
  end

  defp new_gnx_from_nx(array, type, shape, name) do
    {l, c} = get_lines_cols(shape)

    t_charlist = get_type_charlist(type)
    ref = backend().new_gpu_array_from_nx_nif(array, l, c, t_charlist)

    {:nx, type, shape, name, ref}
  end

  defp new_empty_gnx(shape, type) do
    {l, c} = get_lines_cols(shape)

    t_charlist = get_type_charlist(type)
    ref = backend().new_empty_gpu_array_nif(l, c, t_charlist)

    {:nx, type, shape, nil, ref}
  end

  # ------- GNx functions -------

  # === New GNx from existing Nx tensor
  @doc """
  Creates a new GNx (GPU Nx) from an existing Nx tensor.

  ## Parameters

    - `tensor`: An Nx tensor from which to create the GNx.

  ## Returns

    - A GNx with tha same shape and type as the provided Nx tensor, containing the same data but stored in GPU memory.

  """
  def new_gnx(%Nx.Tensor{
        data: data,
        type: type,
        shape: shape,
        names: name
      }) do
    %Nx.BinaryBackend{state: array} = data
    new_gnx_from_nx(array, type, shape, name)
  end

  # === New empty GNx
  @doc """
  Creates a new empty GNx (GPU Nx) with the specified shape and type.

  ## Parameters

    - `shape`: A tuple representing the shape of the GNx (e.g., `{rows, cols}`).
    - `type`: The data type of the GNx (e.g., `{:f, 32}` for 32-bit float).

  ## Returns

    - A new empty GNx with the specified shape and type.

  """
  def new_gnx(shape, type) do
    new_empty_gnx(shape, type)
  end

  # === Get GNx from GPU memory as an Nx tensor in RAM
  @doc """
  Retrieves a GNx (GPU Nx) from GPU memory and as an Nx tensor.

  ## Parameters

    - `gnx`: A GNx to be retrieved.

  ## Returns

    - An Nx tensor with the same shape and type as the provided GNx, containing the data retrieved from GPU memory.

  """
  def get_gnx({:nx, type, shape, name, gnx_ref}) do
    {l, c} = get_lines_cols(shape)
    t_charlist = get_type_charlist(type)

    nx_bin = backend().get_gpu_array_nif(gnx_ref, l, c, t_charlist)

    Nx.from_binary(nx_bin, type) |> Nx.reshape(shape, names: name)
  end

  # ----------------- Nx creation from function (I will re-do this later) -----------------
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
    backend().synchronize_nif()
  end

  @doc """
  Loads the Abstract Syntax Tree (AST) for a given kernel or function used inside a kernel.

  This function tries to extract the module and function name from the provided kernel function reference (assuming to be a kernel).
  If it is a kernel, then the name is extracted this way. If it is a function name, the name is already provided (is the atom itself).

  With the name, a message is sent to the `:module_server` process to request the AST for the specified function.
  The function then waits for a response from the `:module_server` process and returns the AST. If it fails, an error is raised.

  ## Parameters

    - `kernel`: A function reference (e.g., `&Module.function/arity`) representing the kernel function whose AST is to be loaded. Or
    a function name atom (e.g., `:function_name`) representing a function used inside a kernel.

  ## Returns

    - The AST of the specified kernel function.

  ## Raises

    - Raises an error if an unknown message is received from the `:module_server`.
  """
  def load_ast(kernel) do
    # The function may receives a kernel function reference (like `&Module.function/arity`), so we need to extract
    # the module and function name from it.
    # The Macro.escape is used to convert the function reference into a form that can be pattern matched.
    # The pattern matching extracts the module and function name from the function reference.
    {_module, f_name} =
      case Macro.escape(kernel) do
        {:&, [], [{:/, [], [{{:., [], [module, f_name]}, [no_parens: true], []}, _nargs]}]} ->
          {module, f_name}

        # This fallback is used in case we receive a function name directly (for functions used inside kernels).
        f ->
          {:ok, f}
      end

    # Asks the `:module_server` process to get the AST for the specified function name.
    send(:module_server, {:get_ast, f_name, self()})

    # Waits for a response from the `:module_server` process and returns the AST.
    # If an unknown message is received, we raise an error.
    receive do
      {:ast, ast} -> ast
      h -> raise "unknown message from module server #{inspect(h)}"
    end
  end

  # Get the kernel arguments that are not functions
  defp process_args_no_fun([]), do: []

  # Ignoring anonymous function references
  defp process_args_no_fun([{:anon, _lambda_name, _fun_inn_funs} | t1]) do
    process_args_no_fun(t1)
  end

  # GNx: only store the array reference
  defp process_args_no_fun([{:nx, _type, _shape, _name, ref} | t1]) do
    [ref | process_args_no_fun(t1)]
  end

  # Ignoring function references
  defp process_args_no_fun([arg | t1]) when is_function(arg) do
    process_args_no_fun(t1)
  end

  # Everything else is passed as is
  defp process_args_no_fun([arg | t1]) do
    [arg | process_args_no_fun(t1)]
  end

  # ----------------- JIT compilation and kernel spawning -----------------

  @doc """
  Spwans a kernel with JIT compilation.

  Generates the kernel code for the given kernel, compiles it and executes it on the GPU.

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
    initial_delta = JIT.gen_kernel_initial_delta(kast, l)
    # Map of kernel_function_para -> actual_name_in_code
    subs = JIT.get_function_parameters(kast, l)

    kernel_types_and_funs = Map.merge(initial_delta, subs) |> Map.to_list()
    kernel_map_key = {kernel_name, kernel_types_and_funs}

    # ============ temp debug
    IO.inspect(l, label: "provided args (l var)")
    IO.inspect(initial_delta, label: "initial delta")
    IO.inspect(subs, label: "subs")
    IO.inspect(kernel_types_and_funs, label: "kernel_types_and_funs")

    send(:module_server, {:get_kernel, kernel_map_key, self()})

    {kernel_res, types_args} =
      receive do
        # First time launching this kernel with this set of types
        {:kernel, nil} ->
          # Get functions used inside the kernel that are not parameters of the kernel
          fun_graph_asts_sorted =
            JIT.get_non_parameters_func_asts(fun_graph)
            # Now we need to sort these functions in the correct order of inference
            |> JIT.sort_functions_by_call_graph()

          inner_funs_delta = JIT.infer_device_functions_signature(fun_graph_asts_sorted)

          ker_inn_funs_delta = Map.merge(initial_delta, inner_funs_delta)

          # Infers the types of the kernel's variables using the new ker_inn_funs_delta map
          kernel_types_map =
            case JIT.infer_types(kast, ker_inn_funs_delta, kernel_name) do
              {:ok, types} -> types
              {:error, _types, reason} -> raise "Type inference failed: #{reason}"
            end

          # Check if the inferred types contain 'double' or 'tdouble' types since some backends may not
          # support double precision floating point operations (fp64).
          # If the backend implements the `double_supported_nif/0` function, we need to check this.
          if exists_in_backend?(:double_supported_nif, 0) do
            contains_double =
              Map.values(kernel_types_map)
              |> Enum.any?(fn x -> x == :double or x == :tdouble end)

            # If double precision is used, check if the device supports it.
            if contains_double and not backend().double_supported_nif() do
              raise "[PolyHok] Your device does not support double precision floating point operations (fp64). The 'double' data type cannot be used in kernels."
            end
          end

          # Generates kernel string in CUDA/OpenCL/etc
          kernel = JIT.compile_kernel(kast, kernel_types_map, subs)

          # Get a list of tuples {actual_function_param, type} for all formal parameters that are functions.
          param_funs = JIT.get_function_parameters_and_their_types(kast, l, kernel_types_map)

          # Creates a list of tuples where each tuple contains a function name and its inferred type signature
          # These functions were not passed as parameters but were called inside the kernel
          other_funs =
            fun_graph_asts_sorted
            # Creates the tuple {function_name, inferred_type}
            |> Enum.map(fn {x, _ast} -> {x, kernel_types_map[x]} end)
            # Remove functions that could not be inferred
            |> Enum.filter(fn {_, i} -> i != nil end)

          all_funs = other_funs ++ param_funs

          # The JIT.compile_function/2 function compiles the provided function AND it's dependencies (other functions called within
          # a function). To avoid recompiling functions that were already compiled, we provide a MapSet of already compiled functions,
          # so the JIT.compile_function/2 can check and skip a function if necessary.
          # We also re-infer the device functions here now that we have the kernel delta to guarantee we have the correct types
          {comp, _compiled_funs} =
            Enum.reduce(all_funs, {[], MapSet.new()}, fn fun, {code_acc, compiled_funs_acc} ->
              {new_code, compiled_funs_acc} = JIT.compile_function(fun, compiled_funs_acc)
              {code_acc ++ new_code, compiled_funs_acc}
            end)

          includes = JIT.get_includes()
          prog = [includes | comp] ++ [kernel]

          # Concatenating the generated code into a single string
          prog = Enum.reduce(prog, "", fn x, y -> y <> x end)

          debug_logs = Agent.get(:debug_logs_agent, fn state -> state end)

          # Print generated code for debugging purposes if debug logs are enabled
          if debug_logs do
            IO.puts("===== Generated code for kernel '#{kernel_name}' =====")

            # We don't print the includes to reduce clutter
            case comp do
              [] -> IO.puts(kernel)
              l -> IO.puts(Enum.reduce(l, "", fn x, y -> y <> x end) <> kernel)
            end

            IO.puts("==============================================================")
          end

          # List of the inferred types for 'args'
          types_args = JIT.get_types_para(kast, kernel_types_map)

          # Compile the kernel with the JIT compiler and get a reference to the compiled kernel that can be used to launch it
          kernel_res =
            backend().jit_compile_nif(
              Kernel.to_charlist(kernel_name),
              Kernel.to_charlist(prog)
            )

          # We store this compiled kernel reference and it's types_args in the module server, so we can
          # cache it and reuse it in future executions of the same kernel with the same types, avoiding recompilation
          send(:module_server, {:add_kernel, kernel_map_key, {kernel_res, types_args}})

          {kernel_res, types_args}

        {:kernel, {kernel_res, types_args}} ->
          # Kernel was already compiled and cached
          {kernel_res, types_args}
      end

    # 'args' is a list of the actual arguments passed to the kernel, processed to remove any function references
    args = process_args_no_fun(l)

    # Now with the kernel reference and the types of the arguments, we can launch the kernel
    backend().jit_launch_nif(kernel_res, b, t, length(args), types_args, args)

    :ok
  end
end

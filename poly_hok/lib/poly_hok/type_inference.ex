defmodule PolyHok.TypeInference do
  def set_debug_logs(value) do
    Agent.update(:type_inference_debug_logs_agent, fn _old -> value end)
  end

  defp is_debug_logs_enabled?() do
    Agent.get(:type_inference_debug_logs_agent, fn value -> value end)
  end

  defp type_server(global_map) do
    receive do
      {:update_types, key = {_f_name, _initial_delta_list}, new_types} ->
        global_map = Map.put(global_map, key, new_types)
        type_server(global_map)

      {:get_types, key = {_f_name, _initial_delta_list}, caller} ->
        send(caller, {:types_response, Map.get(global_map, key)})
        type_server(global_map)

      _ ->
        IO.puts("Type server received unknown message")
        type_server(global_map)
    end
  end

  @doc """
    Performs type checking and inference on the given AST body using the provided initial type map.
    It recursively infers types until no more types can be inferred.

    ## Parameters
    - map: A map containing initial type information for variables and functions.
    - body: The AST body to perform type inference on.
    - f_name: The name of the function being processed, used for storing and retrieving types from the type server.

    ## Returns
    - A tuple containing a status atom and the final type map after inference. Ex: {:ok, final_map} or {:error, final_map, reason}

  """
  def type_check(map, body, f_name) do
    if Process.whereis(:type_server) == nil do
      ts_pid = spawn_link(fn -> type_server(Map.new()) end)
      Process.register(ts_pid, :type_server)
    end

    logs_en = is_debug_logs_enabled?()

    # The type server keys are in the format {function_name, initial_delta_map_as_list}
    type_server_key = {f_name, Map.to_list(map)}

    if logs_en do
      IO.puts("\n========= [TypeInference] Starting type inference iteration =========")
      IO.puts("[TypeInference] Target function/kernel: #{inspect(f_name)}")
      IO.inspect(map, label: "[TypeInference] Provided initial delta map")

      IO.inspect(type_server_key,
        label: "[TypeInference] Type server key for this function/kernel"
      )
    end

    # Check if the type server already contains a map for this function with this initial delta map. If it does, then it means
    # this function was processed before, so it may contain some already inferred types that we can use for a faster inference!
    # I discovered (in the bad way) that we can't reuse an already inferred type map from a previous iteration if the initial
    # delta map is different, because the initial delta map may contain new information that can change the inference results.
    send(:type_server, {:get_types, type_server_key, self()})

    map =
      receive do
        {:types_response, nil} ->
          map

        {:types_response, types} ->
          if logs_en do
            IO.inspect(types, label: "[TypeInference] Retrieved types map from type server")
          end

          # We need to check if the retrieved type map from the type server is the same as the provided initial delta map
          if types === map do
            # Provided map and type server map are the same, we can use either one
            if logs_en do
              IO.puts(
                "[TypeInference] Retrieved types map from type server is the same as the provided map. Using either one."
              )
            end

            map
          else
            # If the maps are not the same, we merge them
            merged_map = merge_types_map(map, types)

            if logs_en do
              IO.puts(
                "[TypeInference] Retrieved types map from type server is different from the provided map. Merging them to use the most updated info."
              )

              IO.inspect(merged_map, label: "[TypeInference] Merged types map")
            end

            merged_map
          end
      end

    types = infer_types(map, body)
    notinfer = not_infered(Map.to_list(types))

    # Update the type map in the type server process using the same key
    send(:type_server, {:update_types, type_server_key, types})

    if logs_en do
      IO.inspect(types, label: "[TypeInference] Types map after iteration")
      IO.inspect(notinfer, label: "[TypeInference] Not infered")
    end

    if(length(notinfer) > 0) do
      # Second round
      types2 = infer_types(types, body)
      notinfer2 = not_infered(Map.to_list(types2))

      # Save the latest inferred types in the type server
      send(:type_server, {:update_types, type_server_key, types2})

      # Check if something changed
      if length(notinfer) == length(notinfer2) do
        # Return error atom with the latest inferred types and a reason message
        {:error, types2,
         "Could not infer types for the following variables: #{inspect(notinfer2)}"}
      else
        # If something did change, we go for another round
        type_check(types2, body, f_name)
      end
    else
      {:ok, types}
    end
  end

  defp not_infered([]), do: []

  defp not_infered([h | t]) do
    case h do
      {v, :none} ->
        [{v, :none} | not_infered(t)]

      {v, {:none, list}} ->
        [{v, {:none, list}} | not_infered(t)]

      {v, {rt, list}} ->
        fil = Enum.filter(list, fn x -> x == :none end)

        if fil == [] do
          not_infered(t)
        else
          [{v, {rt, list}} | not_infered(t)]
        end

      {_, _} ->
        not_infered(t)
    end
  end

  # This function merges the types map from the type server with the new types map provided as argument to type_check/3.
  # The new types map provided as argument has precedence over the type server map, as it may contains new info.
  defp merge_types_map(arg_map, ts_map) do
    Map.merge(ts_map, arg_map, fn _key, ts_val, arg_val ->
      if ts_val == arg_val do
        # If the types are the same, we can use either one
        ts_val
      else
        # If the values are different, we need to check if one of them is :none,
        # because if one of them is :none, it means that the other one has new info that we can use.
        cond do
          ts_val == :none -> arg_val
          arg_val == :none -> ts_val
          # If both values are different and none of them is :none,
          # it means that we have a conflict in the types. In this case, we chose to use
          # the arg_val, because it is the most updated info (it probably came from the kernel type inference)
          # so it is more likely to be correct than the ts_val, which could lack context information.
          true -> arg_val
        end
      end
    end)
  end

  @doc """
    Adds a return statement to the body of the provided function ONLY IF it returns a value.
  """
  def add_return(body) do
    case body do
      {:do, {:__block__, pos, code}} ->
        {:do, {:__block__, pos, check_return_commands(code)}}

      {:__block__, pos, code} ->
        {:__block__, pos, check_return_commands(code)}

      {:do, exp} ->
        case exp do
          {:return, _, _} ->
            {:do, exp}

          _ ->
            if is_exp?(exp) do
              {:do, {:return, [], [exp]}}
            else
              {:do, check_return_last(exp)}
            end
        end

      {_, _, _} ->
        if is_exp?(body) do
          {:return, [], [body]}
        else
          body
        end
    end
  end

  # When we have a list of commands we need to check only the last one,
  # because only the last command can be a return statement.
  defp check_return_commands(coms) when is_list(coms) do
    # IO.puts("Multiple commands")

    list_len = length(coms)
    coms_with_index = Enum.with_index(coms, 1)

    Enum.map(coms_with_index, fn {com, idx} ->
      if idx == list_len do
        check_return_last(com)
      else
        check_return_command(com)
      end
    end)
  end

  defp check_return_command(com) do
    # IO.puts("Single command - but not last")
    # IO.inspect(com, label: "Command to check")

    case com do
      {:return, _, _} ->
        com

      {:if, info, [exp, [do: do_body]]} ->
        case do_body do
          {:__block__, bl_pos, bl_code} ->
            {:if, info, [exp, [do: {:__block__, bl_pos, check_return_commands(bl_code)}]]}

          do_exp ->
            {:if, info, [exp, [do: check_return_command(do_exp)]]}
        end

      {:if, info, [exp, [do: do_body, else: else_body]]} ->
        processed_do =
          case do_body do
            {:__block__, bl_pos, bl_code} ->
              {:__block__, bl_pos, check_return_commands(bl_code)}

            do_exp ->
              check_return_command(do_exp)
          end

        processed_else =
          case else_body do
            {:__block__, bl_pos, bl_code} ->
              {:__block__, bl_pos, check_return_commands(bl_code)}

            else_exp ->
              check_return_command(else_exp)
          end

        {:if, info, [exp, [do: processed_do, else: processed_else]]}

      _ ->
        com
    end
  end

  defp check_return_last(com) do
    # IO.puts("Checking last command")
    # IO.inspect(com, label: "Command to check")

    case com do
      {:return, _, _} ->
        com

      {:if, info, [exp, [do: do_body]]} ->
        case do_body do
          {:__block__, bl_pos, bl_code} ->
            {:if, info, [exp, [do: {:__block__, bl_pos, check_return_commands(bl_code)}]]}

          do_exp ->
            {:if, info, [exp, [do: check_return_last(do_exp)]]}
        end

      {:if, info, [exp, [do: do_body, else: else_body]]} ->
        processed_do =
          case do_body do
            {:__block__, bl_pos, bl_code} ->
              {:__block__, bl_pos, check_return_commands(bl_code)}

            do_exp ->
              check_return_last(do_exp)
          end

        processed_else =
          case else_body do
            {:__block__, bl_pos, bl_code} ->
              {:__block__, bl_pos, check_return_commands(bl_code)}

            else_exp ->
              check_return_last(else_exp)
          end

        {:if, info, [exp, [do: processed_do, else: processed_else]]}

      _ ->
        if is_exp?(com) do
          {:return, [], [com]}
        else
          com
        end
    end
  end

  defp is_exp?(exp) do
    case exp do
      {{:., _info, [Access, :get]}, _, [_arg1, _arg2]} -> true
      {{:., _, [{_struct, _, nil}, _field]}, _, []} -> true
      {{:., _, [{:__aliases__, _, [_struct]}, _field]}, _, []} -> true
      {op, _info, _args} when op in [:+, :-, :/, :*] -> true
      {op, _info, [_arg1]} when op in [:>>>, :<<<, :~>>, :&&&, :|||, :+++] -> true
      {op, _info, [_arg1, _arg2]} when op in [:<=, :<, :>, :>=, :!=, :==] -> true
      {:!, _info, [_arg]} -> true
      {op, _inf, _args} when op in [:&&, :||] -> true
      {var, _info, nil} when is_atom(var) -> true
      float when is_float(float) -> true
      int when is_integer(int) -> true
      string when is_binary(string) -> true
      _ -> false
    end
  end

  ####################################################### 33

  def infer_types(map, body) do
    # IO.puts "#####"
    # IO.inspect body1
    # body = add_return(map,body1)

    # IO.inspect body
    # IO.puts "####"
    # raise "hell"
    case body do
      {:__block__, _, _code} ->
        infer_block(map, body)

      {:do, {:__block__, pos, code}} ->
        infer_block(map, {:__block__, pos, code})

      {:do, exp} ->
        infer_command(map, exp)

      {_, _, _} ->
        infer_command(map, body)
    end
  end

  defp infer_block(map, {:__block__, _, code}) do
    Enum.reduce(code, map, fn com, acc -> infer_command(acc, com) end)
  end

  defp infer_header_for(map, header) do
    case header do
      {:in, _, [{var, _, nil}, {:range, _, [arg1]}]} ->
        map
        |> Map.put(var, :int)
        |> set_type_exp(:int, arg1)

      {:in, _, [{var, _, nil}, {:range, _, [arg1, arg2]}]} ->
        map
        |> Map.put(var, :int)
        |> set_type_exp(:int, arg1)
        |> set_type_exp(:int, arg2)

      {:in, _, [{var, _, nil}, {:range, _, [arg1, arg2, step]}]} ->
        map
        |> Map.put(var, :int)
        |> set_type_exp(:int, arg1)
        |> set_type_exp(:int, arg2)
        |> set_type_exp(:int, step)
    end
  end

  # The infer_command function is responsible for inferring the types of variables and expressions in a given
  # command (which can be an assignment, a function call, a control structure, etc.).
  # It updates the type map based on the structure of the command and the types of its components.
  defp infer_command(map, code) do
    logs_en = is_debug_logs_enabled?()

    case code do
      {:for, _, [param, [body]]} ->
        map
        |> infer_header_for(param)
        |> infer_types(body)

      {:do_while, _, [[doblock]]} ->
        infer_types(map, doblock)

      {:do_while_test, _, [exp]} ->
        set_type_exp(map, :int, exp)

      {:while, _, [bexp, [body]]} ->
        map
        |> set_type_exp(:int, bexp)
        |> infer_types(body)

      # CRIAÇÃO DE NOVOS VETORES (eu acho)
      {{:., _, [Access, :get]}, _, [arg1, arg2]} ->
        if logs_en do
          IO.inspect(arg1, label: "ic: Array being created (?)")
        end

        array = get_var(arg1)

        map
        |> Map.put(array, :none)
        |> set_type_exp(:int, arg2)

      {shared_atom, _, [{{:., _, [Access, :get]}, _, [arg1, arg2]}]}
      when shared_atom in [:__shared__, :__local] ->
        array = get_var(arg1)

        map
        |> Map.put(array, :none)
        |> set_type_exp(:int, arg2)

      # Assignment to array index
      {:=, _, [{{:., _, [Access, :get]}, _, [{array, _, _}, acc_exp]}, exp]} ->
        if logs_en do
          IO.inspect(array, label: "ic: Array being assigned")
          IO.puts("ic: Array index: #{Macro.to_string(acc_exp)} -> will be set to type int")

          IO.puts(
            "ic: Expression assigning: #{Macro.to_string(exp)} -> will be set to the type of the array elements"
          )
        end

        # Check type of the array being assigned, and set the type of the expression accordingly
        case get_or_insert_var_type(map, array) do
          # If the array has no type yet, we try to infer its type from the expression being assigned to it.
          # If we cannot infer the type, we leave it as :none for now and try to infer it in later iterations.
          {map, :none} ->
            # Get expression type
            type = find_type_exp(map, exp)

            case type do
              :none ->
                map

              # If the expression type is int, we set the array type to :tint and the expression type to :int, and so on
              :int ->
                map
                |> Map.put(array, :tint)
                # Set index type to int always
                |> set_type_exp(:int, acc_exp)
                # Set assigned expression type to int
                |> set_type_exp(:int, exp)

              :float ->
                map
                |> Map.put(array, :tfloat)
                |> set_type_exp(:int, acc_exp)
                |> set_type_exp(:float, exp)

              :double ->
                map
                |> Map.put(array, :tdouble)
                |> set_type_exp(:int, acc_exp)
                |> set_type_exp(:double, exp)
            end

          # Array has type int, float and so on
          {map, :tint} ->
            map
            |> set_type_exp(:int, acc_exp)
            |> set_type_exp(:int, exp)

          {map, :tfloat} ->
            map
            |> set_type_exp(:int, acc_exp)
            |> set_type_exp(:float, exp)

          {map, :tdouble} ->
            map
            |> set_type_exp(:int, acc_exp)
            |> set_type_exp(:double, exp)
        end

      # Assignment to variable
      {:=, _, [var, exp]} ->
        # Get variable name as an atom
        var = get_var(var)

        if logs_en do
          IO.inspect(var, label: "ic: Variable being assigned")
          IO.puts("ic: Expression assigning: #{Macro.to_string(exp)}")
        end

        case get_or_insert_var_type(map, var) do
          # Variable doesn't have a defined type yet. Infer from expression
          {map, :none} ->
            if logs_en do
              IO.puts("ic: Variable #{inspect(var)} has no type yet. Inferring from expression.")
            end

            type_exp = find_type_exp(map, exp)

            if(type_exp != :none) do
              if logs_en do
                IO.puts(
                  "ic: Inferred type #{inspect(type_exp)} for expression #{Macro.to_string(exp)}"
                )

                IO.puts("ic: Setting variable #{inspect(var)} to type #{inspect(type_exp)}.")
              end

              map
              |> Map.put(var, type_exp)
              |> set_type_exp(type_exp, exp)
            else
              if logs_en do
                IO.puts(
                  "ic: Could not infer type for expression #{Macro.to_string(exp)}. Variable #{inspect(var)} will be set to :none for now."
                )
              end

              # If we cannot infer the type from the expression, then probably the variable is being assigned
              # to a function call whose return type is not yet inferred.
              # In this case, we call this function that infers the types of the function parameters.
              map = infer_type_fun(map, exp)

              # Set variable type to :none for now, we'll try to infer it in later iterations
              Map.put(map, var, :none)
            end

          # Variable has a defined type. Set expression to this type.
          {map, var_type} ->
            if logs_en do
              IO.puts(
                "ic: Variable #{inspect(var)} has type #{inspect(var_type)}. Setting expression to #{inspect(var_type)}."
              )
            end

            set_type_exp(map, var_type, exp)
        end

      {:if, _, if_com} ->
        infer_if(map, if_com)

      {:var, _, [{var, _, [{:=, _, [{type, _, nil}, exp]}]}]} ->
        map
        |> Map.put(var, type)
        |> set_type_exp(type, exp)

      {:var, _, [{var, _, [{:=, _, [type, exp]}]}]} ->
        map
        |> Map.put(var, type)
        |> set_type_exp(type, exp)

      {:var, _, [{var, _, [{type, _, _}]}]} ->
        map
        |> Map.put(var, type)

      {:var, _, [{var, _, [type]}]} ->
        map
        |> Map.put(var, type)

      {:type, _, [{var, _, [{type, _, _}]}]} ->
        map
        |> Map.put(var, type)

      {:type, _, [{var, _, [type]}]} ->
        map
        |> Map.put(var, type)

      {:return, _, nil} ->
        Map.put(map, :return, :unit)

      {:return, _, [arg]} ->
        case map[:return] do
          t when t in [nil, :none] ->
            inf_type = find_type_exp(map, arg)

            case inf_type do
              :none ->
                map

              found_type ->
                map = set_type_exp(map, found_type, arg)
                map = Map.put(map, :return, found_type)
                map
            end

          found_type ->
            set_type_exp(map, found_type, arg)
        end

      # Function call command
      {fun, _, args} when is_list(args) ->
        if logs_en do
          IO.puts("ic: Command is a function call: #{inspect(fun)}")
          IO.puts("ic: The return type of function #{inspect(fun)} will be set to :unit (void)")
        end

        type_fun = get_function_type(map, fun)

        if logs_en do
          IO.inspect(type_fun, label: "ic: Type of function #{inspect(fun)} in types map")
        end

        if(type_fun == nil or type_fun == :none) do
          # If the type of the function is (or was) unknown,
          # we infer the types of the arguments and set the function type to :unit (void).
          # Since the function call is not inside an assignment, we can safely assume that it returns nothing (:unit=void)
          {map, infered_types} = infer_types_args(map, args, [])
          Map.put(map, fun, {:unit, infered_types})
        else
          case type_fun do
            {ret, types} ->
              # Set types of function arguments to expected types based on the function type signature,
              # and infer the types of the arguments based on the expected types.
              {map, infered_types} = set_type_args(map, types, args, [])

              # If the function has a return type already defined, it NEEDS to be either :unit or previously set to :none
              # Since we are not inside an assignment, the function return type can't be something else than :unit (void)
              case ret do
                :none ->
                  Map.put(map, fun, {:unit, infered_types})

                :unit ->
                  Map.put(map, fun, {:unit, infered_types})

                t ->
                  raise "Function #{fun} has return type #{t} as is being used in context :unit"
              end
          end
        end

      # Number (wtf?)
      number when is_integer(number) or is_float(number) ->
        raise "Error: a number is not a command"

      # Anything else is ignored (all cases should be covered above, but just in case, we ignore anything that is not recognized as a command)
      {_str, _, _} ->
        IO.puts(
          "ic: Unrecognized command: #{Macro.to_string(code)}. Ignoring it for type inference."
        )

        map
    end
  end

  # Sets the type of a list of expressions based on a list of expected types.
  # Parameters are as follow:
  # - map: The current types map
  # - exp_types: A list of expected types for each expression
  # - args: A list of expressions to set the types for
  # - newtype: A list of the types that were set for each expression (used to return the new types for the function call)
  defp set_type_args(map, [], [], type), do: {map, type}

  defp set_type_args(map, [:none], a1, newtype) when is_tuple(a1) do
    t = find_type_exp(map, a1)

    case t do
      :none ->
        {map, newtype ++ [:none]}

      nt ->
        map = set_type_exp(map, nt, a1)
        {map, newtype ++ [nt]}
    end
  end

  defp set_type_args(map, [t1 | _types], a1, newtype) when is_tuple(a1) do
    map = set_type_exp(map, t1, a1)
    {map, newtype ++ [t1]}
  end

  defp set_type_args(map, [:none | tail], [a1 | args], newtype) do
    t = find_type_exp(map, a1)

    case t do
      :none ->
        set_type_args(map, tail, args, newtype ++ [:none])

      nt ->
        map = set_type_exp(map, nt, a1)
        set_type_args(map, tail, args, newtype ++ [nt])
    end
  end

  defp set_type_args(map, [t1 | types], [a1 | args], newtype) do
    map = set_type_exp(map, t1, a1)
    set_type_args(map, types, args, newtype ++ [t1])
  end

  # Infers the types of a list of expressions.
  # Returns a tuple with the updated map and a list of the inferred types for each expression: {map, [type1, type2, ...]}
  defp infer_types_args(map, [], type), do: {map, type}

  defp infer_types_args(map, [h | tail], type) do
    t = find_type_exp(map, h)

    case t do
      :none ->
        infer_types_args(map, tail, type ++ [:none])

      nt ->
        map = set_type_exp(map, nt, h)
        infer_types_args(map, tail, type ++ [nt])
    end
  end

  # Return the type of the variable or :none if the variable is not declared yet
  # If the variable is not declared, it is added to the map with type :none
  # Returns a tuple with the updated map and the type of the variable: {map, var_type}
  defp get_or_insert_var_type(map, var) do
    var_type = Map.get(map, var)

    if(var_type == nil) do
      map = Map.put(map, var, :none)
      {map, :none}
    else
      {map, var_type}
    end
  end

  # Get the atom representing the variable being accessed
  defp get_var(id) do
    case id do
      # Array access, we return the name of the array
      {{:., _, [Access, :get]}, _, [{array, _, _}, _arg2]} ->
        array

      # Variable access, we return the name of the variable
      {var, _, nil} when is_atom(var) ->
        var
    end
  end

  ################## infering ifs

  defp infer_if(map, [bexp, [do: then]]) do
    map
    |> set_type_exp(:int, bexp)
    |> infer_types(then)
  end

  defp infer_if(map, [bexp, [do: thenbranch, else: elsebranch]]) do
    map
    |> set_type_exp(:int, bexp)
    |> infer_types(thenbranch)
    |> infer_types(elsebranch)
  end

  ###################################################################

  # Set the type of the expression inside the types map, and returns the updated map.
  # It handles different kinds of expressions, such as variable access, array access, binary operations, function calls, etc.
  defp set_type_exp(map, type, exp) do
    logs_en = is_debug_logs_enabled?()

    case exp do
      # Array access: set the array type and the index type
      {{:., info, [Access, :get]}, _, [arg1, arg2]} ->
        case type do
          :int ->
            map
            # Set array type to :tint if the assigned expression type is int, and so on
            |> Map.put(get_var(arg1), :tint)
            # The index must be of type int always
            |> set_type_exp(:int, arg2)

          :float ->
            map
            |> Map.put(get_var(arg1), :tfloat)
            |> set_type_exp(:int, arg2)

          :double ->
            map
            |> Map.put(get_var(arg1), :tdouble)
            |> set_type_exp(:int, arg2)

          _ ->
            raise "Error: location (#{inspect(info)}), invalid type '#{inspect(type)}' for array assignment."
        end

      # Special CUDA structs: nothing to be done, we just return the map
      {{:., _, [{_struct, _, nil}, _field]}, _, []} ->
        map

      {{:., _, [{:__aliases__, _, [_struct]}, _field]}, _, []} ->
        map

      {op, info, [a1, a2]} when op in [:+, :-] and type == :matrex ->
        t1 = find_type_exp(map, a1)
        t2 = find_type_exp(map, a2)

        case t1 do
          :none ->
            case t2 do
              :none -> map
              :int -> set_type_exp(map, :matrex, a1)
              :matrex -> set_type_exp(map, :int, a1)
            end

          :int ->
            map = set_type_exp(map, :int, a1)
            set_type_exp(map, :matrex, a2)

          :matrex ->
            map = set_type_exp(map, :matrex, a1)
            set_type_exp(map, :int, a2)

          tt ->
            raise "Exp #{inspect(a1)} (#{inspect(info)}) has type #{tt} and should have type #{type}"
        end

      {op, _info, args} when op in [:+, :-, :/, :*, :>>>, :<<<, :~>>, :&&&, :|||, :+++] ->
        case args do
          [a1] ->
            set_type_exp(map, type, a1)

          [a1, a2] ->
            t1 = find_type_exp(map, a1)
            t2 = find_type_exp(map, a2)

            case t1 do
              :none ->
                map = set_type_exp(map, type, a1)

                case t2 do
                  :none -> set_type_exp(map, type, a2)
                  _ -> set_type_exp(map, t2, a2)
                end

              _ ->
                map = set_type_exp(map, t1, a1)

                case t2 do
                  :none -> set_type_exp(map, type, a2)
                  _ -> set_type_exp(map, t2, a2)
                end
            end
        end

      {op, info, [arg1, arg2]} when op in [:<=, :<, :>, :>=, :!=, :==] ->
        if(type != :int) do
          raise "Operaotr (#{inspect(op)}) (#{inspect(info)}) is being used in a context #{inspect(type)}"
        end

        t1 = find_type_exp(map, arg1)
        t2 = find_type_exp(map, arg2)

        case t1 do
          :none ->
            case t2 do
              :none ->
                map

              ntype ->
                set_type_exp(map, ntype, arg1)
                set_type_exp(map, ntype, arg2)
            end

          ntype ->
            set_type_exp(map, ntype, arg1)

            case t2 do
              :none ->
                set_type_exp(map, ntype, arg2)

              ntype2 ->
                if ntype != ntype2 do
                  raise "Operator #{inspect(op)} (#{inspect(info)}) is applyed to type #{t1} and type #{t2}."
                else
                  set_type_exp(map, ntype2, arg2)
                end
            end
        end

      {:!, info, [arg]} ->
        if type != :int do
          raise "Operator (!) (#{inspect(info)}) is being used in a context #{inspect(type)}"
        end

        set_type_exp(map, :int, arg)

      {op, inf, args} when op in [:&&, :||] ->
        if(type != :int) do
          raise "Op #{op} (#{inspect(inf)}) is being used in a context: #{inspect(type)}"
        end

        case args do
          [a1] ->
            set_type_exp(map, :int, a1)

          [a1, a2] ->
            map
            |> set_type_exp(:int, a1)
            |> set_type_exp(:int, a2)
        end

      {var, _info, nil} when is_atom(var) ->
        if Map.get(map, var) == nil do
          raise "Error: variable #{inspect(var)} is used in expression before being declared"
        end

        if Map.get(map, var) == :none do
          Map.put(map, var, type)
        else
          if(Map.get(map, var) != type) do
            if type == :int do
              raise "Error: variable #{inspect(var)} should have type integer"
            else
              map
            end
          else
            map
          end
        end

      {fun, _, args} when is_list(args) ->
        type_fun = get_function_type(map, fun)

        if is_special_function?(fun) do
          if logs_en do
            IO.puts(
              "ste: trying to set type for special function #{inspect(fun)}. This function will not be added to types map."
            )
          end

          # For special functions, we don't add them to the types map,
          # we just set the type of their arguments based on the expected types for that function.
          {_ret, expected_types_args} = type_fun

          if length(expected_types_args) != length(args) do
            raise "Special function #{fun} expects #{length(expected_types_args)} arguments, but got #{length(args)}."
          end

          # Set expected types for the arguments of the special function
          {map, _infered_types} = set_type_args(map, expected_types_args, args, [])

          # We return the map without adding the function to the map,
          # we just care about the arguments of the special function being correctly typed
          map
        else
          if(type_fun == nil or type_fun == :none) do
            # If the type of the function is (or was) unknown, we infer the types of the arguments and set the function type
            # to the type provided by the context (which is the type parameter of this function) and the args to their infered types.
            {map, infered_types} = infer_types_args(map, args, [])
            map = Map.put(map, fun, {type, infered_types})
            map
          else
            case type_fun do
              {ret, type_args} ->
                # If the function has a known type, we set the type of the arguments based on the expected argument types provided
                # by the map, and check if the expected return type is compatible with the context (type parameter of this function).
                {map, infered_types} = set_type_args(map, type_args, args, [])

                cond do
                  ret == type ->
                    Map.put(map, fun, {type, infered_types})

                  ret == :none ->
                    Map.put(map, fun, {type, infered_types})

                  true ->
                    raise "Function #{fun} has return type #{ret} and is being used in an #{type} context."
                end

              _ ->
                raise "Error: function #{fun} has type #{type_fun} in the map, but it should have a type signature of the form {return_type, [arg_type1, arg_type2, ...]}"
            end
          end
        end

      {_fun, _, _noargs} ->
        map

      float when is_float(float) ->
        if(type == :float) do
          map
        else
          raise "Type error: #{inspect(float)} is being used in a context of type #{inspect(type)}"
        end

      int when is_integer(int) ->
        if(type == :int || type == :float) do
          map
        else
          raise "Type error: #{inspect(int)} is being used in a context of type #{inspect(type)}"
        end

      string when is_binary(string) ->
        if(type == :string) do
          map
        else
          raise "Type error: #{inspect(string)} is being used in a context of type #{inspect(type)}"
        end
    end
  end

  # Tries to infer the type of a function call.
  # If the type is unknow, it infers the type of the arguments and adds the function to the map with return type :none
  # and the inferred argument types.
  # E.g: fun: {:none, [:int, :float]}
  defp infer_type_fun(map, exp) do
    case exp do
      # Check if the expression is an operation, if it is, we simply return the map unchanged
      {op, _, _args}
      when op in [
             :+,
             :-,
             :/,
             :*,
             :<=,
             :<,
             :>,
             :>=,
             :!=,
             :==,
             :!,
             :&&,
             :||,
             :>>>,
             :<<<,
             :~>>,
             :&&&,
             :|||,
             :+++
           ] ->
        map

      {fun, _, args} when is_list(args) ->
        # Check if the function has a known type in the map
        type_fun = get_function_type(map, fun)

        if(type_fun == nil) do
          # If the type is unknown, we infer the type of the arguments
          {map, infered_types} = infer_types_args(map, args, [])

          # And we add the function to the map with return type :none and the inferred argument types as a list
          map = Map.put(map, fun, {:none, infered_types})

          # Returns the updated map
          map
        else
          # In this case, the function has a known type, but we need to check if it isn't :none
          case type_fun do
            :none ->
              # If the return type is :none, we infer the type of the arguments and update the function type in
              # the map with the inferred argument types and return type :none
              {map, infered_type} = infer_types_args(map, args, [])
              map = Map.put(map, fun, {:none, infered_type})
              map

            {ret, type_args} ->
              # If the return type is known, we set the type of the arguments based on the expected argument types
              # provided by the map, and we return the updated map with the function type expected
              {map, infered_type} = set_type_args(map, type_args, args, [])
              Map.put(map, fun, {ret, infered_type})
          end
        end

      # If the expression is not a function call, we simply ignore it and return the map unchanged
      _ ->
        map
    end
  end

  def infer_type_exp(map, exp) do
    type = find_type_exp(map, exp)

    if type != :none do
      set_type_exp(map, type, exp)
    else
      map
    end
  end

  # Finds and returns the type of a given expression based on the current type map.
  # The difference between this function and infer_command/2 is that this function only returns the type of the
  # expression, while infer_command/2 updates the map with the inferred types.
  # Plus, this function only handles expressions, infer_command/2 handles an entire command, which can
  # contain multiple expressions and other commands.
  defp find_type_exp(map, exp) do
    logs_en = is_debug_logs_enabled?()

    case exp do
      # This is for array access, we return the type of the array element
      {{:., info_, [Access, :get]}, _, [{arg1, _, _}, _arg2]} ->
        case map[arg1] do
          :tint ->
            :int

          :tdouble ->
            :double

          :tfloat ->
            :float

          t when t in [nil, :none] ->
            :none

          ttt ->
            raise "Found invalid type '#{inspect(ttt)}' for array #{inspect(arg1)} (#{inspect(info_)})"
        end

      # This is for struct field access for CUDA constants (threadIdx, blockIdx, etc)
      # We assume that all fields of these structs are integers, but this can be changed if needed
      {{:., _, [{_struct, _, nil}, _field]}, _, []} ->
        :int

      {{:., _, [{:__aliases__, _, [_struct]}, _field]}, _, []} ->
        :int

      # Operations
      {op, info, args} when op in [:+, :-, :/, :*] ->
        case args do
          [a1] ->
            find_type_exp(map, a1)

          [a1, a2] ->
            t1 = find_type_exp(map, a1)
            t2 = find_type_exp(map, a2)

            case t1 do
              :none ->
                t2

              :int ->
                case t2 do
                  :int ->
                    :int

                  :float ->
                    :float

                  :double ->
                    :double

                  t when t in [nil, :none] ->
                    :none

                  _ ->
                    raise "Incompatible operands (#{inspect(info)}: op (#{inspect(op)}) applyed to  type #{inspect(t2)}"
                end

              :float ->
                :float

              :double ->
                :double

              :tfloat ->
                :tfloat

              :tdouble ->
                :tdouble

              :tint ->
                :tint

              _ ->
                raise "Incompatible operands (#{inspect(info)}: op (#{inspect(op)}) applyed to  type #{inspect(t1)}"
            end
        end

      # Bitwise operations must return int
      {op, _, _args} when op in [:>>>, :<<<, :~>>, :&&&, :|||, :+++] ->
        :int

      # Comparison and logical operations, we assume they return int (0 or 1)
      {op, _, _args} when op in [:<=, :<, :>, :>=, :&&, :||, :!, :!=, :==] ->
        :int

      # Variable access, we return the type of the variable or an error if the variable is not declared
      {var, _, nil} when is_atom(var) ->
        case Map.get(map, var) do
          nil ->
            raise "Type Inference error: variable #{inspect(var)} is used in an expression before being declared."

          t ->
            t
        end

      # Function call
      {fun, _, _args} ->
        if logs_en do
          IO.puts("fte: Trying to find type of function call #{inspect(fun)}")
        end

        # Check if the function has a known type
        type_fun = get_function_type(map, fun)

        if logs_en do
          IO.inspect(type_fun, label: "fte: Function #{inspect(fun)} type in map")
        end

        # Returns the return type of the function if it is known, otherwise returns :none
        case type_fun do
          nil -> :none
          :none -> :none
          {ret, _args_types} -> ret
        end

      float when is_float(float) ->
        :float

      int when is_integer(int) ->
        :int

      string when is_binary(string) ->
        :string
    end
  end

  # This function is used to get the type of a function
  # First, it checks if the function being called is a known OpenCL/CUDA function
  # If it is, it returns the known type of the function
  # If it is not, it return the type of the function based on the types map
  defp get_function_type(map, fun) do
    cond do
      is_special_function?(fun) ->
        {PolyHokFunctions.lookup(fun).return_type, PolyHokFunctions.lookup(fun).arg_types}

      true ->
        map[fun]
    end
  end

  @doc """
  This function is used to check if a function is a special OpenCL/CUDA function with known types
  """
  def is_special_function?(fun) do
    PolyHokFunctions.exists?(fun)
  end
end

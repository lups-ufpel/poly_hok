require PolyHok

PolyHok.defmodule Comp do
    
    defmacro gpu_for({:<-,_, [var1, {:..,_, [_b1, e1]}]}, do: body) do
      quote do: Comp.map_coord(  unquote(e1),
                                PolyHok.clo (fn (unquote(var1)) -> (unquote body) end))
      
    end
    defmacro gpu_for({:<-, _ ,[var,tensor]},do: b)  do
      quote do: Comp.map(unquote(tensor), PolyHok.clo (fn (unquote(var)) -> (unquote b) end))
  end
  defmacro gpu_for({:<-,_, [var1, {:..,_, [_b1, e1]}]}, {:<-,_, [var2, {:..,_, [_b2, e2]}]}, do: body) do

    #IO.inspect "Aqui"
    r=      quote do: Comp.comp2xy2D1p(unquote(e1), unquote(e2),
                                       PolyHok.clo (fn (unquote(var1),
                                                    unquote(var2)) -> (unquote body) end))
    #IO.inspect r
    #raise "hell"
    r
end
    def find_return_type_closure({:closure,_name,ast,_free,args}) do
      types_free = JIT.infer_types_actual_parameters(args)
      {ast,_funs} =ast
      {:fn, _, [{:->, _ , [para,_body]}] } = ast
      extra_size = length(para) - length(args)
      extra_types = replicate(extra_size,:none)
      types = extra_types ++ types_free
      delta=JIT.gen_delta_from_type(ast, {:none,types})
      delta=JIT.infer_types(ast,delta)
      r_type = delta[:return]
      case r_type do
        :int -> {:s,32}
        :float -> {:f,32}
        :double -> {:f,64}
        x -> raise "Expression in comprehension has type #{x} which is not suported."
      end
      
     
    end
    defp replicate(0,_x), do: []
    defp replicate(n,v), do: [ v | replicate(n-1,v) ]
    def map(input, f) do
        shape = PolyHok.get_shape(input)
        type = PolyHok.get_type(input)
        result_gpu = PolyHok.new_gnx(shape,type)
        size = Tuple.product(shape)
        threadsPerBlock = 128;
        numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)
    
        PolyHok.spawn(&Comp.map_ker/4,
                  {numberOfBlocks,1,1},
                  {threadsPerBlock,1,1},
                  [input,result_gpu,size, f])
        result_gpu
      end
      defk map_ker(a1,a2,size,f) do
        index = blockIdx.x * blockDim.x + threadIdx.x
        stride = blockDim.x * gridDim.x
  
        for i in range(index,size,stride) do
              a2[i] = f(a1[i])
        end
      end
      def map_coord(size,f) do
        shape = {size}
        #type = PolyHok.get_type(input)
        type = find_return_type_closure(f)
        IO.inspect type
        result_gpu = PolyHok.new_gnx(shape,type)
        
        threadsPerBlock = 128;
        numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)
    
        PolyHok.spawn(&Comp.map_coord_ker/3,
                  {numberOfBlocks,1,1},
                  {threadsPerBlock,1,1},
                  [result_gpu,size, f])
        result_gpu
      end
      defk map_coord_ker(a1,size,f) do
        index = blockIdx.x * blockDim.x + threadIdx.x
        stride = blockDim.x * gridDim.x
  
        for i in range(index,size,stride) do
              a1[i] = f(i)
        end
      end
      defk map2xy2D_kernel(resp,size,f) do
        row  = blockIdx.y * blockDim.y + threadIdx.y
        col = blockIdx.x * blockDim.x + threadIdx.x
      
        if(col < size && row < size) do
          resp[row * size + col] = f(row,col)
        end
      end
      def comp2xy2D1p(size1,size2,f) do
          size = size1
          type = find_return_type_closure(f)
          result_gpu = PolyHok.new_gnx(size1,size2,type)
          
          block_size = 16
         grid_rows = trunc ((size + block_size - 1) / block_size)
        grid_cols = trunc ((size + block_size - 1) / block_size)
      
          PolyHok.spawn(&Comp.map2xy2D_kernel/3,{grid_cols,grid_rows,1},{block_size,block_size,1},[result_gpu,size,f])
      
          r_gpu = PolyHok.get_gnx(result_gpu)
          r_gpu
      end
      
end
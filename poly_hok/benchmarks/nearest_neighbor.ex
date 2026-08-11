require Integer
require PolyHok
defmodule DataSet do
  def open_data_set(file) do
    {:ok, contents} = File.read(file)
    contents
    |> String.split("\n", trim: true)
    |> Enum.map(fn f ->  load_file(f) end)
    |> Enum.concat()
    |> Enum.concat()
 #   |> Enum.unzip()
  end
  def load_file(file) do
    #IO.puts file
    {:ok, contents} = File.read(file)
    contents
    |> String.split("\n", trim: true)
    |> Enum.map(fn line -> words = String.split(line, " ", trim: true)
                           [ elem(Float.parse(Enum.at(words, 6)),0), elem(Float.parse(Enum.at(words,7)), 0) ] end  )
  end
  def gen_data_set_nx_double(n) do
    lat = (7 + Enum.random(0..63)) + :rand.uniform()
    lon = (Enum.random(0..358)) + :rand.uniform()
    acc = <<lat::float-little-64, lon::float-little-64>>
    ref = gen_bin_data_double(n-1, acc)
    %Nx.Tensor{data: %Nx.BinaryBackend{ state: ref}, type: {:f,64}, shape: {n,2}, names:  [nil,nil]}
  end
  defp gen_bin_data_double(0, accumulator), do: accumulator
  defp gen_bin_data_double(size, accumulator)
    do
      lat = (7 + Enum.random(0..63)) + :rand.uniform()
      lon = (Enum.random(0..358)) + :rand.uniform()
      gen_bin_data_double(
        size - 1,
        <<accumulator::binary, lat::float-little-64, lon::float-little-64>>
      )
    end
  def gen_data_set_nx(n) do
    lat = (7 + Enum.random(0..63)) + :rand.uniform()
    lon = (Enum.random(0..358)) + :rand.uniform()
    acc = <<lat::float-little-32, lon::float-little-32>>
    ref = gen_bin_data(n-1, acc)
    %Nx.Tensor{data: %Nx.BinaryBackend{ state: ref}, type: {:f,32}, shape: {n,2}, names:  [nil,nil]}
  end
  defp gen_bin_data(0, accumulator), do: accumulator
  defp gen_bin_data(size, accumulator)
    do
      lat = (7 + Enum.random(0..63)) + :rand.uniform()
      lon = (Enum.random(0..358)) + :rand.uniform()
      gen_bin_data(
        size - 1,
        <<accumulator::binary, lat::float-little-32, lon::float-little-32>>
      )
    end
  def gen_data_set(n), do: gen_data_set_(n,[])
  def gen_data_set_(0,data), do: data
  def gen_data_set_(n,data) do
    lat = (7 + Enum.random(0..63)) + :rand.uniform();
      lon = (Enum.random(0..358)) + :rand.uniform();
      gen_data_set_(n-1, [lat,lon|data])

  end
  def gen_lat_long(_l,c) do
    if(Integer.is_even(c)) do
      (Enum.random(0..358)) + :rand.uniform()
    else
      (7 + Enum.random(0..63)) + :rand.uniform()
    end
  end
end


PolyHok.defmodule NN do
  include CAS_Double
  def euclid_seq(l,lat,lng), do: euclid_seq_(l,lat,lng,[])
  def euclid_seq_([m_lat,m_lng|array],lat,lng,data) do
    # m_lat = Enum.at(array,0)
     #m_lng = Enum.at(array,1)

     value = :math.sqrt((lat-m_lat)*(lat-m_lat)+(lng-m_lng)*(lng-m_lng))
     #value = :math.sqrt((lat-m_lat)*(lat-m_lat)+(lng-m_lng)*(lng-m_lng))
     euclid_seq_(array,lat,lng,[value|data])
  end
  def euclid_seq_([],_lat,_lng, data) do
    data
  end
  def reduce(ref, acc, f) do

    {l,c} = PolyHok.get_shape_gnx(ref)
    type = PolyHok.get_type_gnx(ref)
    size = l*c
     result_gpu  = PolyHok.new_gnx(Nx.tensor([[acc]] , type: type))

     threadsPerBlock = 256
     blocksPerGrid = div(size + threadsPerBlock - 1, threadsPerBlock)
     numberOfBlocks = blocksPerGrid
     PolyHok.spawn(&NN.reduce_kernel/4,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[ref, result_gpu, f, size])
     result_gpu
 end
 defk reduce_kernel(a, ref4, f,n) do

   __shared__ cache[256]

   tid = threadIdx.x + blockIdx.x * blockDim.x;
   cacheIndex = threadIdx.x

   temp = ref4[0]

   while (tid < n) do
     temp = f(a[tid], temp)
     tid = blockDim.x * gridDim.x + tid
   end

   cache[cacheIndex] = temp
     __syncthreads()

   i = blockDim.x/2

   while (i != 0 ) do  ###&& tid < n) do
     #tid = blockDim.x * gridDim.x + tid
     if (cacheIndex < i) do
       cache[cacheIndex] = f(cache[cacheIndex + i] , cache[cacheIndex])
     end

   __syncthreads()
   i = i/2
   end

 if (cacheIndex == 0) do
   current_value = ref4[0]
   while(!(current_value == atomic_cas(ref4,current_value,f(cache[0],current_value)))) do
     current_value = ref4[0]
   end
 end

 end
  defk map_step_2para_1resp_kernel(d_array, d_result, step,  par1, par2,size,f) do


    #var globalId int = blockDim.x * ( gridDim.x * blockIdx.y + blockIdx.x ) + threadIdx.x
    globalId = threadIdx.x + blockIdx.x * blockDim.x
    id  = step * globalId
    #f(id,id)
    if (globalId < size) do
      d_result[globalId] = f(d_array+id, par1,par2)
    end
  end
  def map_step_2para_1resp(d_array,step, par1, par2, size, f) do
    type = PolyHok.get_type_gnx(d_array)

      distances_device = PolyHok.new_gnx(1,size, type)
      PolyHok.spawn(&NN.map_step_2para_1resp_kernel/7,{size,1,1},{1,1,1},[d_array,distances_device,step,par1,par2,size,f])
      distances_device
  end
  defd euclid(d_locations, lat, lng) do
    return sqrt((lat-d_locations[0])*(lat-d_locations[0])+(lng-d_locations[1])*(lng-d_locations[1]))
      #return sqrt((lat-d_locations[0])*(lat-d_locations[0])+(lng-d_locations[1])*(lng-d_locations[1]))
    end

  defd menor(x,y) do
    if (x<y) do
      x
     else
       y
     end
    end
end


[arg] = System.argv()

size = String.to_integer(arg)

data_set_host = DataSet.gen_data_set_nx_double(size)

#data_set_host = Nx.tensor(DataSet.gen_data_set(size),  type: {:f,32} )

#IO.inspect data_set_host

#IO.inspect data_set_host

prev = System.monotonic_time()

_r= PolyHok.new_gnx(data_set_host)
|> NN.map_step_2para_1resp(2,0.0,0.0,size, &NN.euclid/3)
|> NN.reduce(50000.0,&NN.menor/2)
|> PolyHok.get_gnx
#      |> IO.inspect



next = System.monotonic_time()
IO.puts "PolyHok\t#{size}\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"

#result_elixir = Enum.reverse(NN.euclid_seq(list_data_set,0.0,0.0))



#IO.puts("NN = #{nn[1]}")


#IO.inspect (Enum.reduce(result_elixir,0, fn (x,y)-> if y == 0 do x else if x<y do x else y end end end))

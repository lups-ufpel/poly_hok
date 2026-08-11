require PolyHok
use Comp

[arg] = System.argv()
size = String.to_integer(arg)

#host_a = Enum.to_list(1..size) |> Nx.tensor(type: :f64)
#host_b = Enum.to_list(1..size) |> Nx.tensor(type: :f64)

host_a = PolyHok.new_nx_from_function(1,size,{:f,64},fn -> 1.0 end) #:rand.uniform(1000) end)
host_b = PolyHok.new_nx_from_function(1,size,{:f,64},fn -> 1.0 end) #:rand.uniform(1000) end)
prev = System.monotonic_time()

a = PolyHok.new_gnx(host_a)
b = PolyHok.new_gnx(host_b)

result = (Comp.gpu_for i <- 0..size, do:  2 * a[i] + b[i]) |> PolyHok.get_gnx

next = System.monotonic_time()

IO.inspect result

IO.puts "PolyHok\t#{size}\t#{System.convert_time_unit(next-prev,:native,:millisecond)} "

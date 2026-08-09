[arg] = System.argv()
m = String.to_integer(arg)

#values_per_pixel = 4

#prev = System.monotonic_time()

#result_gpu = PolyHok.new_gnx(dim*dim,4,{:s,32})

mat1 = PolyHok.new_nx_from_function(m,m,{:f,32},fn -> :rand.uniform(1000) end)

IO.inspect mat1
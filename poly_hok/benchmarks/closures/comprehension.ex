require PolyHok
use Comp


dev_vet = PolyHok.new_gnx(Nx.tensor([1,2,3,4,5,6,7,8,9,10]))

v = 3

(Comp.gpu_for n <- dev_vet,  do: v * n)
|> PolyHok.get_gnx
|> IO.inspect


a = PolyHok.new_gnx(Nx.tensor(Enum.to_list(1..1000),type: {:s, 32}))
b = PolyHok.new_gnx(Nx.tensor(Enum.to_list(1..1000),type: {:s, 32}))

size = 1000

(Comp.gpu_for i <- 0..size, do:  2 * a[i] + b[i])
|> PolyHok.get_gnx
|> IO.inspect

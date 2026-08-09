require PolyHok
use Ske


dev_vet = PolyHok.new_gnx(Nx.tensor([1,2,3,4,5,6,7,8,9,10]))

x = 1

fun = PolyHok.clo fn y -> x + y end 

host_vet = dev_vet
|> Ske.map(fun)
|> PolyHok.get_gnx

IO.inspect host_vet
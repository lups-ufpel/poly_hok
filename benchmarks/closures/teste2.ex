require PolyHok
use Ske

defmodule Teste do
def teste()  do
  x = 2
  PolyHok.clo fn y -> x + y end 
end
end

dev_vet = PolyHok.new_gnx(Nx.tensor([1,2,3,4,5,6,7,8,9,10]))

fun = Teste.teste() 

host_vet = dev_vet
|> Ske.map(fun)
|> PolyHok.get_gnx

IO.inspect host_vet
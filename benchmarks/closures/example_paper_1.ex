require PolyHok
use Ske


dev_arr = PolyHok.new_gnx(Nx.tensor(Enum.to_list(1..1000), type: :s32))

x = 10

fun = PolyHok.clo fn y -> x + y end 

host_arr = dev_arr
|> Ske.map(fun)
|> PolyHok.get_gnx

IO.inspect host_arr
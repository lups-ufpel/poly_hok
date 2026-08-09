require PolyHok
use Ske

n = 1000

arr1 = Nx.tensor([Enum.to_list(1..n)],type: {:s, 32})


arr1
    |> PolyHok.new_gnx
    |> Ske.map(PolyHok.phok fn (x) -> x + 1 end)
    |> PolyHok.get_gnx
    |> IO.inspect

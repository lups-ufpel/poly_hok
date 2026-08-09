require PolyHok
defmodule BMP do
  @on_load :load_nifs
  def load_nifs do
      :erlang.load_nif("./priv/bmp_nifs", 0)
  end
  def gen_bmp_int_nif(_string,_dim,_mat) do
      raise "gen_bmp_nif not implemented"
  end
  def gen_bmp_float_nif(_string,_dim,_mat) do
    raise "gen_bmp_nif not implemented"
end
  def gen_bmp_int(string,dim,%Nx.Tensor{data: data, type: _type, shape: _shape, names: _name}) do
    %Nx.BinaryBackend{ state: array} = data
    gen_bmp_int_nif(string,dim,array)
  end
  def gen_bmp_float(string,dim,%Nx.Tensor{data: data, type: _type, shape: _shape, names: _name}) do
    %Nx.BinaryBackend{ state: array} = data
    gen_bmp_float_nif(string,dim,array)
  end
end

PolyHok.defmodule Julia do
  defd julia(x,y,dim) do
    scale  = 0.1
    jx = scale * (dim - x)/dim
    jy = scale * (dim - y)/dim

    cr  = -0.8
    ci  = 0.156
    ar  = jx
    ai  = jy
    for i in range(0,200) do
        nar = (ar*ar - ai*ai) + cr
        nai = (ai*ar + ar*ai) + ci
        if ((nar * nar)+(nai * nai ) > 1000.0) do
          return 0
        end
        ar = nar
        ai = nai
    end
    return 1
  end
  defd julia_function(ptr,dim,x,y) do

    juliaValue = julia(x,y,dim)

    ptr[0] = 255 * juliaValue;
    ptr[1] = 0;
    ptr[2] = 0;
    ptr[3] = 255;
  end



end

use Ske

[arg] = System.argv()
m = String.to_integer(arg)

dim = m

#values_per_pixel = 4

result_gpu = PolyHok.new_gnx({dim,dim,4},{:s,32})

prev = System.monotonic_time()

image = result_gpu
  |> Ske.map(&Julia.julia_function/4,[dim], return: false, dim: :two, coord: true)
  |> PolyHok.get_gnx

next = System.monotonic_time()

IO.puts "PolyHok\t#{dim}\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"

BMP.gen_bmp_int(~c"juliaske.bmp",dim,image)

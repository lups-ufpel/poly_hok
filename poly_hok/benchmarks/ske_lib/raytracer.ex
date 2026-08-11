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

PolyHok.defmodule RayTracer do


defd raytracing(image, width,  spheres ,x,y) do

  ox = 0.0
  oy = 0.0
  ox = (x - width/2)
  oy = (y - width/2)

  r = 0.0
  g = 0.0
  b = 0.0

  maxz = -99999.0

  for i in range(0, 20) do

    sphereRadius = spheres[i * 7 + 3]

    dx = ox - spheres[i * 7 + 4]
    dy = oy - spheres[i * 7 + 5]
    n = 0.0
    t = -99999.0
    dz = 0.0
    if (dx * dx + dy * dy) <  (sphereRadius * sphereRadius) do
      dz = sqrtf(sphereRadius * sphereRadius - (dx * dx) - (dy * dy))
      n = dz / sqrtf(sphereRadius * sphereRadius)
      t = dz + spheres[i * 7 + 6]
    else
      t = -99999.0
      n = 0.0
    end

    if t > maxz do
      fscale = n
      r = spheres[i * 7 + 0] * fscale
      g = spheres[i * 7 + 1] * fscale
      b = spheres[i * 7 + 2] * fscale
      maxz = t
    end
  end

  image[0] = r * 255
  image[1] = g * 255
  image[2] = b * 255
  image[3] = 255

end

end
defmodule Main do
    def rnd(x) do
        :rand.uniform() *x
        #x * Random.randint(1, 32767) / 32767
    end
    def sphereMaker2(0,_dim), do: []
    def sphereMaker2(n,dim) do
      [
        Main.rnd(1),
        Main.rnd(1),
        Main.rnd(1),
        Main.rnd(trunc(dim/10)) + (dim/50),
        Main.rnd( dim ) - trunc(dim/2),
        Main.rnd( dim ) - trunc(dim/2),
        Main.rnd( dim ) - trunc(dim/2)
        | sphereMaker2(n - 1,dim)]

    end
    def dim do
      {d, _} = Integer.parse(Enum.at(System.argv, 0))
      d
    end
    def spheres do
     # {s, _} = Integer.parse(Enum.at(System.argv, 1))
     # s
     20
    end

    def main do
        use Ske

        sphereList = Nx.tensor([sphereMaker2(Main.spheres,Main.dim)], type: {:f,32})

        width = Main.dim
        height = width



        prev = System.monotonic_time()

        ref_sphere = PolyHok.new_gnx(sphereList)
        ref_image = PolyHok.new_gnx({width, height, 4},{:s,32})

        Ske.map(ref_image, &RayTracer.raytracing/5 , [width, ref_sphere], dim: :two, return: false, coord: true)
      
        image = PolyHok.get_gnx(ref_image)

        next = System.monotonic_time()
        IO.puts "PolyHok\t#{width}\t#{System.convert_time_unit(next-prev,:native,:millisecond)} "


        BMP.gen_bmp_int(~c"ray.bmp",width,image)



    end
end

Main.main

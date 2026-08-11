require PolyHok
import Bitwise

#Random.seed(313)

defmodule BMP do
  @on_load :load_nifs
  def load_nifs do
      :erlang.load_nif('./priv/bmp_nifs', 0)
  end
  def gen_bmp_int_nif(_string,_dim,_mat) do
      raise "gen_bmp_nif not implemented"
  end
  def gen_bmp_float_nif(_string,_dim,_mat) do
    raise "gen_bmp_nif not implemented"
end
  def gen_bmp_int(string,dim,%Nx.Tensor{data: data, type: type, shape: shape, names: name}) do
    %Nx.BinaryBackend{ state: array} = data
    gen_bmp_int_nif(string,dim,array)
  end
  def gen_bmp_float(string,dim,%Nx.Tensor{data: data, type: type, shape: shape, names: name}) do
    %Nx.BinaryBackend{ state: array} = data
    gen_bmp_float_nif(string,dim,array)
  end
end

PolyHok.defmodule RayTracer do


defk raytracing(width, height, spheres, image) do

  x = threadIdx.x + blockIdx.x * blockDim.x
  y = threadIdx.y + blockIdx.y * blockDim.y
  offset = x + y * blockDim.x * gridDim.x

  ox = 0.0
  oy = 0.0
  ox = (x - width/2)
  oy = (y - height/2)

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

  image[offset * 4 + 0] = r * 255
  image[offset * 4 + 1] = g * 255
  image[offset * 4 + 2] = b * 255
  image[offset * 4 + 3] = 255

end
end


defmodule Bmpgen do
  def fileHeaderSize do #constant
    14
  end

  def infoHeaderSize do #constant
    40
  end

  def bytes_per_pixel do
    4
  end
  def recursiveWrite([]) do
    IO.puts("done!")
  end
  def recursiveWrite([0.0, 0.0, 0.0, 0.0 | _rest]) do
    IO.puts("rest of the list is empty. Finalizing write.")
  end

  #def recursiveWrite([a | image], i, max) do
  def recursiveWrite([r, g, b, 255.0 | image]) do
    l = [<<trunc(g)>>, <<trunc(b)>>, <<trunc(r)>>, <<255>>]
    File.write!("img-gpuraytracer-#{Main.dim}x#{Main.dim}.bmp", l, [:append])
    recursiveWrite(image)
  end

  def writeFileHeader(height, stride) do
    fileSize = Bmpgen.fileHeaderSize + Bmpgen.infoHeaderSize + (stride * height)
    fileHeader = ['B'] ++ ['M'] ++ [<<fileSize>>] ++ [<<fileSize >>> 8>>] ++ [<<fileSize >>> 16>>] ++ [<<fileSize >>> 24>>] ++ List.duplicate(<<0>>, 4) ++ [<<Bmpgen.fileHeaderSize + Bmpgen.infoHeaderSize>>] ++ List.duplicate(<<0>>, 3)
    IO.puts("\n-----------------------\n")
    File.write!("img-gpuraytracer-#{Main.dim}x#{Main.dim}.bmp", fileHeader)
  end

  def writeInfoHeader(height, width) do
    infoHeader = [<<Bmpgen.infoHeaderSize>>] ++ List.duplicate(<<0>>, 3) ++ [<<width>>, <<width >>> 8>>, <<width >>> 16>>, <<width >>> 24>>, <<height>>, <<height >>> 8>>, <<height >>> 16>>, <<height >>> 24>>, <<1>>, <<0>>, <<Bmpgen.bytes_per_pixel * 8>>] ++ List.duplicate(<<0>>, 25)
    File.write!("img-gpuraytracer-#{Main.dim}x#{Main.dim}.bmp", infoHeader, [:append])
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

    def spherePrinter([]) do
      File.write!("spheregpu.txt", "done\n", [:append])

    end
    def spherePrinter([ r, g, b, _radius, _x, _y, _z | list]) do
      File.write!("spheregpu.txt", "\t r: #{r}", [:append])
      File.write!("spheregpu.txt", "\t g: #{g}", [:append])
      File.write!("spheregpu.txt", "\t b: #{b}", [:append])
      File.write!("spheregpu.txt", "\n", [:append])
      spherePrinter(list)
    end


    def sphereMaker(spheres, max, max) do
      max = max - 1
        Matrex.set(spheres, 1, max * 7 + 1, Main.rnd(1))
        |> Matrex.set( 1, max * 7 + 2, Main.rnd(1)) #g
        |> Matrex.set( 1, max * 7 + 3, Main.rnd(1)) #b
        |> Matrex.set( 1, max * 7 + 4, Main.rnd(20) + 5) #radius
        |> Matrex.set( 1, max * 7 + 5, Main.rnd(Main.dim) - Main.dim/2) #x
        |> Matrex.set( 1, max * 7 + 6, Main.rnd(Main.dim) - Main.dim/2) #y
        |> Matrex.set( 1, max * 7 + 7, Main.rnd(256) - 128) #z
    end
    def sphereMaker(spheres, n, max) do

      Matrex.set(spheres, 1, n * 7 + 1, Main.rnd(1)) #r
      |> Matrex.set( 1, (n - 1) * 7 + 2, Main.rnd(1)) #g
      |> Matrex.set( 1, (n - 1) * 7 + 3, Main.rnd(1)) #b
      |> Matrex.set( 1, (n - 1) * 7 + 4, Main.rnd(20) + 5) #radius
      |> Matrex.set( 1, (n - 1) * 7 + 5, Main.rnd(Main.dim) - Main.dim/2) #x
      |> Matrex.set( 1, (n - 1) * 7 + 6, Main.rnd(Main.dim) - Main.dim/2) #y
      |> Matrex.set( 1, (n - 1) * 7 + 7, Main.rnd(256) - 128) #z
      |> sphereMaker(n + 1, max)
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

        sphereList = Nx.tensor([sphereMaker2(Main.spheres,Main.dim)], type: {:f,32})

        width = Main.dim
        height = width


        #imageList = Matrex.zeros(1, (width + 1) * (height + 1) * 4)

        prev = System.monotonic_time()

        refSphere = PolyHok.new_gnx(sphereList)
        refImag = PolyHok.new_gnx(1,width * height  * 4,{:s,32})

        PolyHok.spawn_jit(&RayTracer.raytracing/4,{trunc(width/16),trunc(height/16),1},{16,16,1},[width, height, refSphere, refImag])

        image = PolyHok.get_gnx(refImag)

        next = System.monotonic_time()
        IO.puts "PolyHok\t#{width}\t#{System.convert_time_unit(next-prev,:native,:millisecond)} "


        BMP.gen_bmp_int('ray.bmp',width,image)

        #image = Matrex.to_list(image)

        #widthInBytes = width * Bmpgen.bytes_per_pixel
        #paddingSize = rem((4 - rem(widthInBytes, 4)), 4)
        #stride = widthInBytes + paddingSize

        #Bmpgen.writeFileHeader(height, stride)
        #Bmpgen.writeInfoHeader(height, width)
        #Bmpgen.recursiveWrite(image)


    end
end

Main.main

# Bmp

This is small Elixir library for generating BMP images from raw pixel data. The data must be stored in an Nx tensor of type `int` or `float`. The tensor must have 4 channels (BGRA) and the pixel values must be in the range [0, 255] to be rendered properly.

## Usage

To use the library, you must add it as a dependency in your project by adding the following line to your `mix.exs` file:

```elixir
defp deps do
  [
    {:bmp, git: "https://github.com/lups-ufpel/poly_hok.git", sparse: "tools/bmp"}
  ]
end
```

## Example

To create a red square of 50x50 pixels in the center of a black square of 100x100 pixels, you can use the following code:

```elixir
dim = 100

# Creating a black square of size 100x100 with 4 channels (BGRA)
image = Nx.broadcast(0, {dim * dim, 4})

red = Nx.tensor([0, 0, 255, 255], type: :s32)  # Red color in BGRA

# Adding 50 red pixels from (25, 25) up to (25, 75)
image = Enum.reduce(25..75, image,
  fn line, img ->
    Nx.put_slice(img, [line * dim + 25, 0], Nx.broadcast(red, {50, 4}))
  end
)

# Save the image to a file
Bmp.gen_bmp_int("red_square.bmp", dim, image)

```

This code is available in the `test/red_square.exs` file of this repository. You can run it with the command:

```bash
mix run test/red_square.exs
```

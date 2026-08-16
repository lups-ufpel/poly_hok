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

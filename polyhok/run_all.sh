echo "dot product double"
mix run benchmarks/dot_product_double.ex 1000
echo "dot product"
mix run benchmarks/dot_product.ex 1000
echo "julia"
mix run benchmarks/julia.ex 1024
echo "mm"
mix run benchmarks/mm.ex 1024
echo "nbodies"
mix run benchmarks/nbodies.ex 1000
echo "nearest neightbor"
mix run benchmarks/nearest_neighbor.ex 1000
echo "raytracer"
mix run benchmarks/raytracer.ex 1000
echo "ske lib dot product"
mix run benchmarks/ske_lib/dot_product.ex 1000
echo "ske lib julia"
mix run benchmarks/ske_lib/julia.ex 1024
echo "ske lib raytracer"
mix run benchmarks/ske_lib/raytracer.ex 1024
IO.inspect(quote do
  1 <<< 2
end, label: "Testing operators")

IO.inspect(quote do
  1 >>> 2
end, label: "Testing operators")

IO.inspect(quote do
  1 &&& 2
end, label: "Testing operators")

IO.inspect(quote do
  1 ||| 2
end, label: "Testing operators")

IO.inspect(quote do
  1 ~>> 2
end, label: "Testing operators")

IO.inspect(quote do
  1 +++ 2
end, label: "Testing operators")
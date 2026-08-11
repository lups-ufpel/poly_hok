defmodule CudaBackendTest do
  use ExUnit.Case
  doctest CudaBackend

  test "greets the world" do
    assert CudaBackend.hello() == :world
  end
end

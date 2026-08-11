defmodule OpenclBackendTest do
  use ExUnit.Case
  doctest OpenclBackend

  test "greets the world" do
    assert OpenclBackend.hello() == :world
  end
end

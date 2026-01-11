defmodule MargarineTest do
  use ExUnit.Case
  doctest Margarine

  test "greets the world" do
    assert Margarine.hello() == :world
  end
end

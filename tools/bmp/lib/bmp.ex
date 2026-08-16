defmodule Bmp do
  @on_load :load_nifs
  def load_nifs do
    nif_path = Application.app_dir(:bmp, "priv/bmp_nifs") |> to_charlist()
    ret = :erlang.load_nif(nif_path, 0)

    case ret do
      :ok ->
        :ok

      {:error, {reason, text}} ->
        IO.puts("[BMP] Failed to load NIF")
        IO.puts("[BMP] Reason: #{inspect(reason)}")
        IO.puts("[BMP] Text: #{text}")

        :erlang.halt(1)
    end
  end

  def gen_bmp_int(file_name, dim, %Nx.Tensor{data: data}) do
    %Nx.BinaryBackend{state: array} = data

    file_name_c = file_name |> to_charlist()
    gen_bmp_int_nif(file_name_c, dim, array)

    :ok
  end

  def gen_bmp_float(file_name, dim, %Nx.Tensor{data: data}) do
    %Nx.BinaryBackend{state: array} = data

    file_name_c = file_name |> to_charlist()
    gen_bmp_float_nif(file_name_c, dim, array)

    :ok
  end

  def test_bmp_generation(file_name, dim) do
    file_name_c = file_name |> to_charlist()

    test_bmp_generation_nif(file_name_c, dim)

    :ok
  end

  defp gen_bmp_int_nif(_file_name, _dim, _mat) do
    :erlang.nif_error(:nif_not_loaded)
  end

  defp gen_bmp_float_nif(_file_name, _dim, _mat) do
    :erlang.nif_error(:nif_not_loaded)
  end

  defp test_bmp_generation_nif(_file_name, _dim) do
    :erlang.nif_error(:nif_not_loaded)
  end
end

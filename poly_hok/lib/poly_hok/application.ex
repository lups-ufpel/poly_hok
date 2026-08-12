defmodule PolyHok.Application do
  use Application

  @impl true
  def start(_type, _args) do
    case Application.fetch_env(:poly_hok, :backend) do
      {:ok, backend} ->
        # Saving the backend in Erlang's persistent storage. It is extremely fast to read from.
        :persistent_term.put({PolyHok, :backend}, backend)

      :error ->
        raise "PolyHok: backend not configured. Please set the backend in your config/runtime.exs file."
    end

    children = [
      %{
        id: :debug_logs_agent,
        start: {Agent, :start_link, [fn -> false end, [name: :debug_logs_agent]]}
      },
      %{
        id: :type_inference_debug_logs_agent,
        start: {Agent, :start_link, [fn -> false end, [name: :type_inference_debug_logs_agent]]}
      }
    ]

    opts = [strategy: :one_for_one, name: PolyHok.Supervisor]
    Supervisor.start_link(children, opts)
  end
end

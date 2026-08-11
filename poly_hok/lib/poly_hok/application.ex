defmodule PolyHok.Application do
  use Application

  @impl true
  def start(_type, _args) do
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

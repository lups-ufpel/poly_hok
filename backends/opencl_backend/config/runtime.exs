import Config

# This is used for local development and testing. Elixir ignores config/runtime.exs when pulling dependencies.
# The users have to set their own backend in the config/runtime.exs file. This is intentional behavior.
config :poly_hok, backend: OpenclBackend

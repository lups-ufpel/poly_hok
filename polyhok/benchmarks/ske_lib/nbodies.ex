require PolyHok

PolyHok.defmodule NBodies do

  defd gpu_nBodies(p,c,n) do
    softening = 0.000000001
    dt = 0.01
    fx = 0.0
    fy = 0.0
    fz = 0.0
    for j in range(0,n) do
        dx = c[6*j] - p[0];
        dy = c[6*j+1] - p[1];
        dz = c[6*j+2] - p[2];
        distSqr = dx*dx + dy*dy + dz*dz + softening;
        invDist = 1.0/sqrt(distSqr);
        invDist3  = invDist * invDist * invDist;

        fx = fx + dx * invDist3;
        fy = fy + dy * invDist3;
        fz = fz + dz * invDist3;
      end
  p[3] = p[3]+ dt*fx;
  p[4] = p[4]+ dt*fy;
  p[5] = p[5]+ dt*fz;

  end
  defd gpu_integrate(p, dt, n) do
      p[0] = p[0] + p[3]*dt;
      p[1] = p[1] + p[4]*dt;
      p[2] = p[2] + p[5]*dt;

  end
end

use Ske

[arg] = System.argv()

user_value = String.to_integer(arg)



nBodies = user_value #3000;
size_body = 6

h_buf = PolyHok.new_nx_from_function(nBodies,size_body,{:f,64},fn -> :rand.uniform() end )
prev = System.monotonic_time()

d_buf = PolyHok.new_gnx(h_buf)

_gpu_resp = d_buf
  |> Ske.map(&NBodies.gpu_nBodies/3, [d_buf,nBodies], return: false)
  |> Ske.map(&NBodies.gpu_integrate/3, [0.01,nBodies], return: false)
  |> PolyHok.get_gnx
  #|> IO.inspect

  next = System.monotonic_time()

IO.puts "PolyHok\t#{user_value}\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"

#IO.inspect gpu_resp

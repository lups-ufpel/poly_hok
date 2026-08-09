require PolyHok

PolyHok.defmodule NBodies do

  defk map_step_no_resp_kernel(d_array,  step, size,f) do


    globalId  = blockDim.x * ( gridDim.x * blockIdx.y + blockIdx.x ) + threadIdx.x
    id  = step * globalId
    #f(id,id)
    if (globalId < size) do
      f(d_array+id)
    end
  end
  def map_no_resp(d_array,  size, f) do
    block_size =  128;
    {_l,step} = PolyHok.get_shape_gnx(d_array)
    nBlocks = floor ((size + block_size - 1) / block_size)

      PolyHok.spawn(&NBodies.map_step_no_resp_kernel/4,{nBlocks,1,1},{block_size,1,1},[d_array,step,size,f])
      d_array
  end
 
end


[arg] = System.argv()

user_value = String.to_integer(arg)

nBodies = user_value #3000;
size_body = 6



h_buf = PolyHok.new_nx_from_function(nBodies,size_body,{:f,64},fn -> :rand.uniform() end )

#h_buf = PolyHok.new_nx_from_function(nBodies,size_body,{:f,32},fn -> 1 end )

#IO.inspect h_buf

prev = System.monotonic_time()

d_buf = PolyHok.new_gnx(h_buf)

c = d_buf
n = nBodies

n_bodies_fun = PolyHok.clo fn p ->
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

dt = 0.01

integrate_fun = PolyHok.clo fn p ->
    p[0] = p[0] + p[3]*dt;
    p[1] = p[1] + p[4]*dt;
    p[2] = p[2] + p[5]*dt;
end

_gpu_resp = d_buf
  |> NBodies.map_no_resp(nBodies, n_bodies_fun)
  |> NBodies.map_no_resp(nBodies, integrate_fun)
  |> PolyHok.get_gnx
  #|> IO.inspect

  next = System.monotonic_time()

IO.puts "PolyHok\t#{user_value}\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"


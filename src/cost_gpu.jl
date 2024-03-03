#///////////////////////////////////////
#// File Name: cost_gpu.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/01/15
#// Description: GPU (CUDA) Cost model for Risk Sensitive Stochastic SAC
#///////////////////////////////////////

using CUDA
using LinearAlgebra

# Instantaneous Cost
# # Position
function kernel_instant_position_cost!(out::AbstractArray{Float32, 2},
                                       ex_array::AbstractArray{Float32, 3},
                                       target_pos_array::AbstractArray{Float32, 2},
                                       Cep::AbstractArray{Float32, 2})
    # out : (num_controls, T)
    # ex_array : (num_controls, T, 4)
    # target_pos_array : (T, 2)
    # Cep : (2, 2)

    ii = (blockIdx().x - 1)*blockDim().x + threadIdx().x; # dimension for nominal control candidates
    jj = (blockIdx().y - 1)*blockDim().y + threadIdx().y; # dimension for timesteps

    if (ii <= size(out, 1)) && (jj <= size(out, 2))
        position_error_x = ex_array[ii, jj, 1] - target_pos_array[jj, 1];
        position_error_y = ex_array[ii, jj, 2] - target_pos_array[jj, 2];
        out[ii, jj] = 0.5f0*(position_error_x*Cep[1, 1]*position_error_x +
                             position_error_x*Cep[1, 2]*position_error_y +
                             position_error_y*Cep[2, 1]*position_error_x +
                             position_error_y*Cep[2, 2]*position_error_y);
    end
    return nothing
end

function instant_position_cost(ex_array::AbstractArray{Float32, 3},
                               target_pos_array::AbstractArray{Float32, 2},
                               param::CostParameter;
                               threads::NTuple{2, Int}=(8, 32))

    # out : (num_controls, total_timesteps - 1)
    # ex_array : (num_controls, total_timesteps, 4)
    # target_pos_array : (total_timesteps, 2)

    Cep = cu(param.Cep);
    out = CuArray{Float32, 2}(undef, size(ex_array, 1), size(ex_array, 2) - 1)
    threads = threads;
    numblocks_x = ceil(Int, size(out, 1)/threads[1]);
    numblocks_y = ceil(Int, size(out, 2)/threads[2]);
    blocks = (numblocks_x, numblocks_y)
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks kernel_instant_position_cost!(out, ex_array[:, 1:end-1, :], target_pos_array[1:end-1, :], Cep)
    end
    return out
end

# # Control
function kernel_instant_control_cost!(out::AbstractArray{Float32, 2},
                                      u_array::AbstractArray{Float32, 3},
                                      Cu::AbstractArray{Float32, 2})
    # out : (num_controls, T)
    # u_array : (num_controls, T, 2)
    # Cu : (2, 2)

    ii = (blockIdx().x - 1)*blockDim().x + threadIdx().x; # dimension for nominal control candidates
    jj = (blockIdx().y - 1)*blockDim().y + threadIdx().y; # dimension for timesteps

    if (ii <= size(out, 1)) && (jj <= size(out, 2))
        out[ii, jj] = 0.5f0*(u_array[ii, jj, 1]*Cu[1, 1]*u_array[ii, jj, 1] +
                             u_array[ii, jj, 1]*Cu[1, 2]*u_array[ii, jj, 2] +
                             u_array[ii, jj, 2]*Cu[2, 1]*u_array[ii, jj, 1] +
                             u_array[ii, jj, 2]*Cu[2, 2]*u_array[ii, jj, 2])
    end
    return nothing
end

function instant_control_cost(u_array::AbstractArray{Float32, 3},
                              param::CostParameter;
                              threads::NTuple{2, Int}=(8, 32))

    # u_array : (num_controls, total_timesteps - 1)
    # out : (num_controls, total_timesteps - 1)

    Cu = cu(param.Cu);
    out = CuArray{Float32, 2}(undef, size(u_array, 1), size(u_array, 2))
    threads = threads;
    numblocks_x = ceil(Int, size(out, 1)/threads[1]);
    numblocks_y = ceil(Int, size(out, 2)/threads[2]);
    blocks = (numblocks_x, numblocks_y)
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks kernel_instant_control_cost!(out, u_array, Cu)
    end
    return out
end

# # Collision
function kernel_instant_collision_cost!(out::AbstractArray{Float32, 3},
                                        ex_array::AbstractArray{Float32, 3},
                                        ap_array::AbstractArray{Float32, 4},
                                        time_idx_ap_array::AbstractArray{Int32, 1},
                                        control_idx_ex_array::AbstractArray{Int32, 1},
                                        α_col::Float32,
                                        λ_col::Float32)

    # out : (num_samples*num_controls, T, num_ado_agents)
    # ex_array : (num_controls, T, 4)
    # ap_array : (num_samples*num_controls, prediction_steps + 1, num_ado_agents, 2)
    # time_idx_ap_array : (T)
    # control_idx_ex_array : (num_samples*num_controls)

    ii = (blockIdx().x - 1)*blockDim().x + threadIdx().x; # dimension for samples (and nominal control candidates)
    jj = (blockIdx().y - 1)*blockDim().y + threadIdx().y; # dimension for timesteps
    kk = (blockIdx().z - 1)*blockDim().z + threadIdx().z; # dimension for ado agents

    if (ii <= size(out, 1)) && (jj <= size(out, 2)) && (kk <= size(out, 3))
        ll = control_idx_ex_array[ii];
        mm = time_idx_ap_array[jj];

        squared_dist = (ex_array[ll, jj, 1] - ap_array[ii, mm, kk, 1])^2 +
                       (ex_array[ll, jj, 2] - ap_array[ii, mm, kk, 2])^2;
        out[ii, jj, kk] = α_col*CUDA.exp(-1/(2*λ_col)*squared_dist);
    end
    return nothing
end

function instant_collision_cost(ex_array::AbstractArray{Float32, 3},
                                ap_array::AbstractArray{Float32, 4},
                                time_idx_ap_array::AbstractArray{Int32, 1},
                                control_idx_ex_array::AbstractArray{Int32, 1},
                                param::CostParameter;
                                threads::NTuple{3, Int}=(8, 4, 2))

    # out : (num_samples*num_controls, total_timesteps - 1, num_ado_agents)
    # ex_array : (num_controls, total_timesteps, 4)
    # ap_array : (num_samples*num_controls, prediction_steps + 1, num_ado_agents, 2)
    # time_idx_ap_array : (total_timesteps)
    # control_idx_ex_array : (num_samples*num_controls)

    α_col = Float32(param.α_col);
    λ_col = Float32(param.λ_col);
    out = CuArray{Float32, 3}(undef, size(ap_array, 1), size(ex_array, 2) - 1, size(ap_array, 3));
    if size(out, 3) == 0 # If no ado agents exist, return all zeros.
        return cu(zeros(size(out)));
    end
    threads = threads;
    numblocks_x = ceil(Int, size(out, 1)/threads[1]);
    numblocks_y = ceil(Int, size(out, 2)/threads[2]);
    numblocks_z = ceil(Int, size(out, 3)/threads[3]);
    blocks = (numblocks_x, numblocks_y, numblocks_z);
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks kernel_instant_collision_cost!(out, ex_array[:, 1:end-1, :],
                                                                           ap_array,
                                                                           time_idx_ap_array[1:end-1],
                                                                           control_idx_ex_array,
                                                                           α_col, λ_col);
    end
    return out
end

# Terminal Cost
# # Position
function kernel_terminal_position_cost!(out::AbstractArray{Float32, 2},
                                        ex_array::AbstractArray{Float32, 3},
                                        target_pos_array::AbstractArray{Float32, 2},
                                        Cep::AbstractArray{Float32, 2})

    # out : (num_controls, 1)
    # ex_array : (num_controls, 1, 4)
    # target_pos_array : (1, 2)
    # Cep : (2, 2)

    kernel_instant_position_cost!(out, ex_array, target_pos_array, Cep)
    return nothing
end

function terminal_position_cost(ex_array::AbstractArray{Float32, 3},
                                target_pos_array::AbstractArray{Float32, 2},
                                param::CostParameter;
                                threads::NTuple{2, Int}=(256, 1))

    # out : (num_controls, 1)
    # ex_array : (num_controls, total_timesteps, 4)
    # target_pos_array : (total_timesteps, 2)

    Cep = cu(param.β_pos*param.Cep);
    out = CuArray{Float32, 2}(undef, size(ex_array, 1), 1)
    threads = threads;
    numblocks_x = ceil(Int, size(out, 1)/threads[1]);
    numblocks_y = ceil(Int, size(out, 2)/threads[2]);
    blocks = (numblocks_x, numblocks_y)
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks kernel_terminal_position_cost!(out, ex_array[:, end:end, :], target_pos_array[end:end, :], Cep)
    end
    return out
end

# # Collision
function kernel_terminal_collision_cost!(out::AbstractArray{Float32, 3},
                                         ex_array::AbstractArray{Float32, 3},
                                         ap_array::AbstractArray{Float32, 4},
                                         time_idx_ap_array::AbstractArray{Int32, 1},
                                         control_idx_ex_array::AbstractArray{Int32, 1},
                                         α_col::Float32,
                                         λ_col::Float32)

    # out : (num_samples*num_controls, 1, num_ado_agents)
    # ex_array : (num_controls, 1, 4)
    # ap_array : (num_samples*num_controls, prediction_steps + 1, num_ado_agents, 2)
    # time_idx_ap_array : (1)
    # control_idx_ex_array : (num_samples*num_controls)

    kernel_instant_collision_cost!(out, ex_array, ap_array,
                                   time_idx_ap_array, control_idx_ex_array,
                                   α_col, λ_col)
    return nothing
end

function terminal_collision_cost(ex_array::AbstractArray{Float32, 3},
                                 ap_array::AbstractArray{Float32, 4},
                                 time_idx_ap_array::AbstractArray{Int32, 1},
                                 control_idx_ex_array::AbstractArray{Int32, 1},
                                 param::CostParameter;
                                 threads::NTuple{3, Int}=(64, 1, 4))

    # out : (num_samples*num_controls, 1, num_ado_agents)
    # ex_array : (num_controls, 1, 4)
    # ap_array : (num_samples*num_controls, prediction_steps + 1, num_ado_agents, 2)
    # time_idx_ap_array : (1)
    # control_idx_ex_array : (num_samples*num_controls)

    α_col = Float32(param.β_col*param.α_col);
    λ_col = Float32(param.λ_col);
    out = CuArray{Float32, 3}(undef, size(ap_array, 1), 1, size(ap_array, 3));
    if size(out, 3) == 0 # If no ado agents exist, return all zeros.
        return cu(zeros(size(out)));
    end
    threads = threads;
    numblocks_x = ceil(Int, size(out, 1)/threads[1]);
    numblocks_y = ceil(Int, size(out, 2)/threads[2]);
    numblocks_z = ceil(Int, size(out, 3)/threads[3]);
    blocks = (numblocks_x, numblocks_y, numblocks_z);
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks kernel_terminal_collision_cost!(out, ex_array[:, end:end, :],
                                                                            ap_array,
                                                                            time_idx_ap_array[end:end],
                                                                            control_idx_ex_array,
                                                                            α_col, λ_col)
    end
    return out
end

# Instantaneous Cost Gradient (with respect to ego robot state)
# # Position
function kernel_instant_position_cost_gradient!(out::AbstractArray{Float32, 2},
                                                ex_array::AbstractArray{Float32, 2},
                                                target_pos_array::AbstractArray{Float32, 2},
                                                Cep::AbstractArray{Float32, 2})
    # out : (T, 4)
    # ex_array : (T, 4)
    # target_pos_array : (T, 2)
    # Cep : (2, 2)

    ii = (blockIdx().x - 1)*blockDim().x + threadIdx().x; # dimension for timesteps

    if (ii <= size(out, 1))
        position_error_x = ex_array[ii, 1] - target_pos_array[ii, 1];
        position_error_y = ex_array[ii, 2] - target_pos_array[ii, 2];
        out[ii, 1] = Cep[1, 1]*position_error_x + Cep[1, 2]*position_error_y;
        out[ii, 2] = Cep[2, 1]*position_error_x + Cep[2, 2]*position_error_y;
        out[ii, 3] = 0.0f0;
        out[ii, 4] = 0.0f0;
    end
    return nothing
end

function instant_position_cost_gradient(ex_array::AbstractArray{Float32, 2},
                                        target_pos_array::AbstractArray{Float32, 2},
                                        param::CostParameter;
                                        threads::Int=64)

    # out : (total_timesteps - 1, 4)
    # ex_array : (total_timesteps, 4)
    # target_pos_array : (total_timesteps, 2)

    Cep = cu(param.Cep);
    out = CuArray{Float32, 2}(undef, size(ex_array, 1) - 1, 4)
    threads = threads;
    blocks = ceil(Int, size(out, 1)/threads);
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks kernel_instant_position_cost_gradient!(out, ex_array[1:end-1, :], target_pos_array[1:end-1, :], Cep)
    end
    return out
end

## Collision
function kernel_instant_collision_cost_gradient!(out::AbstractArray{Float32, 4},
                                                 ex_array::AbstractArray{Float32, 2},
                                                 ap_array::AbstractArray{Float32, 4},
                                                 time_idx_ap_array::AbstractArray{Int32, 1},
                                                 α_col::Float32,
                                                 λ_col::Float32)

    # out : (num_samples, T, num_ado_agents, 4)
    # ex_array : (T, 4)
    # ap_array : (num_samples, prediction_steps + 1, num_ado_agents, 2)
    # time_idx_ap_array : (T)

    ii = (blockIdx().x - 1)*blockDim().x + threadIdx().x; # dimension for samples
    jj = (blockIdx().y - 1)*blockDim().y + threadIdx().y; # dimension for timesteps
    kk = (blockIdx().z - 1)*blockDim().z + threadIdx().z; # dimension for ado agents

    if (ii <= size(out, 1)) && (jj <= size(out, 2)) && (kk <= size(out, 3))
        mm = time_idx_ap_array[jj];

        squared_dist = (ex_array[jj, 1] - ap_array[ii, mm, kk, 1])^2 +
                       (ex_array[jj, 2] - ap_array[ii, mm, kk, 2])^2;
        collision_cost = α_col*CUDA.exp(-1/(2*λ_col)*squared_dist);
        out[ii, jj, kk, 1] =
                    collision_cost/λ_col*(ap_array[ii, mm, kk, 1] - ex_array[jj, 1]);
        out[ii, jj, kk, 2] =
                    collision_cost/λ_col*(ap_array[ii, mm, kk, 2] - ex_array[jj, 2]);
        out[ii, jj, kk, 3] = 0.0f0;
        out[ii, jj, kk, 4] = 0.0f0;
    end
    return nothing
end

function instant_collision_cost_gradient(ex_array::AbstractArray{Float32, 2},
                                         ap_array::AbstractArray{Float32, 4},
                                         time_idx_ap_array::AbstractArray{Int32, 1},
                                         param::CostParameter;
                                         threads::NTuple{3, Int}=(32, 4, 2))

    # out : (num_samples, total_timesteps - 1, num_ado_agents, 4)
    # ex_array : (total_timesteps, 4)
    # ap_array : (num_samples, prediction_steps + 1, num_ado_agents, 2)
    # time_idx_ap_array : (total_timesteps)

    α_col = Float32(param.α_col);
    λ_col = Float32(param.λ_col);
    out = CuArray{Float32, 4}(undef, size(ap_array, 1), size(ex_array, 1) - 1, size(ap_array, 3), 4);
    if size(out, 3) == 0 # If no ado agents exist, return all zeros.
        return cu(zeros(size(out)));
    end
    threads = threads;
    numblocks_x = ceil(Int, size(out, 1)/threads[1]);
    numblocks_y = ceil(Int, size(out, 2)/threads[2]);
    numblocks_z = ceil(Int, size(out, 3)/threads[3]);
    blocks = (numblocks_x, numblocks_y, numblocks_z);
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks kernel_instant_collision_cost_gradient!(out, ex_array[1:end-1, :],
                                                                                    ap_array,
                                                                                    time_idx_ap_array[1:end-1],
                                                                                    α_col, λ_col);
    end
    return out
end

# Terminal Cost Gradient (with respect to ego robot state)
# # Position
function kernel_terminal_position_cost_gradient!(out::AbstractArray{Float32, 2},
                                                 ex_array::AbstractArray{Float32, 2},
                                                 target_pos_array::AbstractArray{Float32, 2},
                                                 Cep::AbstractArray{Float32, 2})
    # out : (1, 4)
    # ex_array : (1, 4)
    # target_pos_array : (1, 2)
    # Cep : (2, 2)

    kernel_instant_position_cost_gradient!(out, ex_array, target_pos_array, Cep)
    return nothing
end

function terminal_position_cost_gradient(ex_array::AbstractArray{Float32, 2},
                                         target_pos_array::AbstractArray{Float32, 2},
                                         param::CostParameter;
                                         threads::Int=1)

    # out : (1, 4)
    # ex_array : (1, 4)
    # target_pos_array : (1, 2)

    Cep = cu(param.β_pos*param.Cep);
    out = CuArray{Float32, 2}(undef, 1, 4)
    threads = threads;
    blocks = ceil(Int, size(out, 1)/threads);
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks kernel_terminal_position_cost_gradient!(out, ex_array[end:end, :], target_pos_array[end:end, :], Cep)
    end
    return out
end

# # Collision
function kernel_terminal_collision_cost_gradient!(out::AbstractArray{Float32, 4},
                                                  ex_array::AbstractArray{Float32, 2},
                                                  ap_array::AbstractArray{Float32, 4},
                                                  time_idx_ap_array::AbstractArray{Int32, 1},
                                                  α_col::Float32,
                                                  λ_col::Float32)

    # out : (num_samples, 1, num_ado_agents, 4)
    # ex_array : (1, 4)
    # ap_array : (num_samples, prediction_steps + 1, num_ado_agents, 2)
    # time_idx_ap_array : (1)

    kernel_instant_collision_cost_gradient!(out, ex_array, ap_array, time_idx_ap_array, α_col, λ_col)
    return nothing
end

function terminal_collision_cost_gradient(ex_array::AbstractArray{Float32, 2},
                                          ap_array::AbstractArray{Float32, 4},
                                          time_idx_ap_array::AbstractArray{Int32, 1},
                                          param::CostParameter;
                                          threads::NTuple{3, Int}=(64, 1, 4))

    # out : (num_samples, 1, num_ado_agents, 4)
    # ex_array : (1, 4)
    # ap_array : (num_samples, prediction_steps + 1, num_ado_agents, 2)
    # time_idx_ap_array : (1)

    α_col = Float32(param.β_col*param.α_col);
    λ_col = Float32(param.λ_col);
    out = CuArray{Float32, 4}(undef, size(ap_array, 1), 1, size(ap_array, 3), 4);
    if size(out, 3) == 0 # If no ado agents exist, return all zeros.
        return cu(zeros(size(out)));
    end
    threads = threads;
    numblocks_x = ceil(Int, size(out, 1)/threads[1]);
    numblocks_y = ceil(Int, size(out, 2)/threads[2]);
    numblocks_z = ceil(Int, size(out, 3)/threads[3]);
    blocks = (numblocks_x, numblocks_y, numblocks_z);
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks kernel_terminal_collision_cost_gradient!(out, ex_array[end:end, :],
                                                                                     ap_array,
                                                                                     time_idx_ap_array[end:end],
                                                                                     α_col, λ_col);
    end
    return out
end

#///////////////////////////////////////
#// File Name: forward_backward_simulation.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/01/29
#// Description: Forward-backward simulation model for Risk Sensitive SAC
#///////////////////////////////////////

using CUDA
using LinearAlgebra
using PyCall
using Random
using RobotOS
import StatsFuns: logsumexp

# Simulation Parameter
struct SimulationParameter <: Parameter
    dtc::Float64                   # Time Interval for euler integral
    dtr::Float64                   # Time Interval for Discrete-time Dynamics
    dto::Float64                   # Time Interval for Trajectron Prediction
    prediction_steps::Int64        # Prediction steps for ado robots
    num_samples::Int64             # Number of samples (per ado) for ado simulation
    cost_param::DRCCostParameter
end

function SimulationParameter(traj_scene_loader::TrajectronSceneLoader,
                             traj_predictor::TrajectronPredictor,
                             dtc::Float64, cost_param::DRCCostParameter)
    dto = traj_scene_loader.dto;
    prediction_steps = traj_predictor.param.prediction_steps;
    num_samples = traj_predictor.param.num_samples;
    return SimulationParameter(dtc, 0.1, dto, prediction_steps, num_samples,
                               cost_param)
end

function SimulationParameter(traj_scene_loader::TrajectronSceneLoader,
                             oracle_predictor::OraclePredictor,
                             dtc::Float64, cost_param::DRCCostParameter)
    dto = traj_scene_loader.dto;
    prediction_steps = oracle_predictor.param.prediction_steps;
    num_samples = 1;
    return SimulationParameter(dtc, 0.1, dto, prediction_steps, num_samples,
                               cost_param)
end

function SimulationParameter(gaussian_predictor::GaussianPredictor,
                             dtc::Float64, cost_param::DRCCostParameter)
    dto = gaussian_predictor.dto;
    prediction_steps = gaussian_predictor.param.prediction_steps;
    num_samples = gaussian_predictor.param.num_samples;
    return SimulationParameter(dtc, 0.1, dto, prediction_steps, num_samples,
                               cost_param)
end

# Simulated Cost Result
mutable struct SimulationCostResult
    inst_cnt_cost_array_gpu::CuArray{Float32, 2} # instantaneous control costs: (num_controls, total_timesteps-1)
    inst_pos_cost_array_gpu::CuArray{Float32, 2} # instantaneous position costs: (num_controls, total_timesteps - 1)
    inst_col_cost_array_gpu::CuArray{Float32, 3} # instantaneous collision costs: (num_samples*num_controls, total_timesteps-1, num_ado_agents)

    term_pos_cost_array_gpu::CuArray{Float32, 2} # terminal position costs: (num_controls, 1)
    term_col_cost_array_gpu::CuArray{Float32, 3} # terminal collision costs: (num_samples*num_controls, 1, num_ado_agents)
end

# Simulated Cost Gradient Result
mutable struct SimulationCostGradResult
    inst_pos_cost_grad_array_gpu::CuArray{Float32, 2} # instantaneous position cost gradients: (total_timesteps-1, 4)
    inst_col_cost_grad_array_gpu::CuArray{Float32, 4} # instantaneous collision cost gradients: (num_samples, total_timesteps-1, num_ado_agents, 4)

    term_pos_cost_grad_array_gpu::CuArray{Float32, 2} # terminal position cost gradients: (1, 4) array
    term_col_cost_grad_array_gpu::CuArray{Float32, 4} # terminal collision cost gradients: (num_samples, 1, num_ado_agents, 4)
end

# Simulation Result
mutable struct SimulationResult
    u_nominal_array::Vector{Vector{Float64}} # best nominal control schedule to be used
    u_nominal_idx::Int64 # best nominal control idx
    e_state_array::Vector{RobotState} # simulation of ego robot states
    measurement_schedule::Vector{Time} # measurement schedule
    prediction_dict::Dict{String, Array{Float64, 3}} # predicted ado states (lower time resolution)
    ap_array_gpu::CuArray{Float32, 4} # predicted ado states (higher time resolution, concatenated)
    ado_ids_array::Vector{String} # correspondence between 1st dim of expanded_ap_array and ado agent id
    sampled_total_costs::Vector{Float64} # sampled total costs
    risk_val::Float64 # risk value (i.e. our objective)
    e_costate_array::Array{Float64, 3} # (4, total_timesteps, num_samples) array
    e_costate_array_constraint::Union{Nothing, Array{Float64, 3}} # nothing or (4, constraint_timesteps, num_samples) array
    sampled_total_costs_constraint::Union{Nothing, Vector{Float64}} # nothing or sampled total costs for constraint computation
    risk_constraint::Union{Nothing, Float64} # nothing or risk value for constraint computation
end


# Helper Functions
# # convert array of u_array to 3-dimensional u_array
function process_u_arrays(u_arrays::Vector{Vector{Vector{Float64}}})
    # u_arrays: array of length num_controls of arrays of length total_timesteps-1 of arrays of lenght 2
    num_controls = length(u_arrays)
    total_timesteps = length(u_arrays[1]) + 1
    @assert all([length(u_array) == total_timesteps - 1 for u_array in u_arrays])

    u_array = Array{Float64, 3}(undef, num_controls, total_timesteps - 1, 2)
    for ii = 1:num_controls
        @inbounds u_array[ii, :, :] = permutedims(hcat(u_arrays[ii]...), [2, 1]);
    end
    return u_array
end

# # Forward Simulation
function simulate_forward(e_init::RobotState,
                          u_array::Vector{Vector{Float64}},
                          sim_param::SimulationParameter)
    e_state_array = Vector{RobotState}(undef, length(u_array) + 1);
    e_state_array[1] = e_init;
    for ii in eachindex(u_array)
        @inbounds e_state_array[ii + 1] = transition(e_state_array[ii],
                                                     u_array[ii],
                                                     sim_param.dtc)
    end
    return e_state_array
end
# # Forward Simulation (CUDA version)
function simulate_forward(e_init::RobotState,
                          u_array_gpu::CuArray{Float32, 3}, # multiple nominal control version
                          sim_param::SimulationParameter)
    # ex_init: 4-length vector of initial ego state.
    # u_array_gpu: (num_controls, total_timesteps-1, 2) array of acceleration inputs
    out_vel = similar(u_array_gpu)
    dtc = Float32(sim_param.dtc);
    # compute velocity (Euler integration)
    CUDA.accumulate!(+, out_vel, u_array_gpu.*sim_param.dtc, dims=2);
    out_vel = cat(cu(zeros(size(u_array_gpu, 1), 1, 2)), out_vel, dims=2);
    ev_init = get_velocity(e_init);
    out_vel[:, :, 1] .+= Float32(ev_init[1]);
    out_vel[:, :, 2] .+= Float32(ev_init[2]);

    # compute position (Euler integration)
    out_pos = similar(u_array_gpu)
    CUDA.accumulate!(+, out_pos, out_vel[:, 1:end-1, :].*dtc, dims=2);
    out_pos = cat(cu(zeros(size(u_array_gpu, 1), 1, 2)), out_pos, dims=2);
    ep_init = get_position(e_init);
    out_pos[:, :, 1] .+= Float32(ep_init[1]);
    out_pos[:, :, 2] .+= Float32(ep_init[2]);

    # (num_controls, total_timesteps, 4) array
    return cat(out_pos, out_vel, dims=3)
end

# # Measurement Schedule Handler
function get_measurement_schedule(w_init::WorldState, horizon::Float64,
                                  sim_param::SimulationParameter)
    measurement_schedule =
        Vector{Time}(undef, Int64(round(horizon/sim_param.dto, digits=6)));
    # If no measurement has been taken, assume the initial measurement time of
    # Time(a*dto) where a is the minimum integer so that a*dto > w_init.t
    if w_init.t_last_m == nothing
        a = 0;
        while a*sim_param.dto <= to_sec(w_init.t)
            a += 1;
        end
        measurement_schedule[1] = Time(a*sim_param.dto);
    else
        measurement_schedule[1] =
            w_init.t_last_m + Duration(sim_param.dto);
    end
    for ii = 2:length(measurement_schedule)
        measurement_schedule[ii] =
            measurement_schedule[ii - 1] + Duration(sim_param.dto);
    end
    return measurement_schedule
end
function get_measurement_schedule(w_init::WorldState,
                                  sim_param::SimulationParameter)
    measurement_schedule = Vector{Time}(undef, sim_param.prediction_steps);
    # If no measurement has been taken, assume the initial measurement time of
    # Time(a*dto) where a is the minimum integer so that a*dto > w_init.t
    if w_init.t_last_m == nothing
        a = 0;
        while a*sim_param.dto <= to_sec(w_init.t)
            a += 1;
        end
        measurement_schedule[1] = Time(a*sim_param.dto);
    else
        measurement_schedule[1] =
            w_init.t_last_m + Duration(sim_param.dto);
    end
    for ii = 2:length(measurement_schedule)
        measurement_schedule[ii] =
            measurement_schedule[ii - 1] + Duration(sim_param.dto);
    end
    return measurement_schedule
end

function get_target_pos_array(ex_array_gpu::AbstractArray{Float32, 3},
                              w_init::WorldState,
                              target_trajectory::Trajectory2D,
                              sim_param::SimulationParameter)

    # target_pos_aray_gpu : (total_timesteps, 2)
    # ex_array_gpu : (num_controls, total_timesteps, 4)

    t_array = [w_init.t]
    for ii = 1:size(ex_array_gpu, 2) - 1
        push!(t_array, t_array[end] + Duration(sim_param.dtc))
    end

    target_pos_array = Array(transpose(hcat(map(t -> get_position(target_trajectory, t), t_array)...)));
    return target_pos_array # (total_timesteps, 2) array
end

function process_ap_dict(ex_array_gpu::CuArray{Float32, 3},
                         w_init::WorldState,
                         measurement_schedule::Vector{Time},
                         prediction_dict::Dict{String, Array{Float64, 3}},
                         sim_param::SimulationParameter)

    # prediction_dict : each value is (num_samples*num_controls, prediction_steps, 2) array
    # ap_array : (num_samples*num_controls, prediction_steps + 1, num_ado_agents, 2) array
    # time_idx_ap_array : (total_timesteps) array of time indices for referencing ap_array
    # control_idx_ex_array : (num_samples*num_controls) array of nominal control indices for referencing ex_array
    # ado_ids_array : (num_ado_agents) array of ado ids

    # ratio between dto and dtc
    expansion_factor = Int64(sim_param.dto/sim_param.dtc);
    # number of nominal control candidates
    num_controls = size(ex_array_gpu, 1);

    # total simulation timesteps
    total_timesteps = size(ex_array_gpu, 2);
    # timesteps between current and first measurement time
    init_timesteps =
        Int64(to_nsec(measurement_schedule[1] - w_init.t)/
              (sim_param.dtc*1e9));
    # timesteps between final measurement time and final e_state time
    remaining_timesteps = total_timesteps -
                          (init_timesteps + expansion_factor*(sim_param.prediction_steps - 1));

    # time_idx_ap_array
    time_idx_ap_array = Vector{Int64}(undef, total_timesteps);
    time_idx_ap_array[1:init_timesteps] .= 1;
    time_idx_ap_array[init_timesteps+1:end-remaining_timesteps] = repeat(2:1:sim_param.prediction_steps,
                                                                         inner=expansion_factor);
    time_idx_ap_array[end-remaining_timesteps+1:end] .= sim_param.prediction_steps + 1;

    # control_idx_ex_array
    control_idx_ex_array = repeat(1:1:num_controls, inner=sim_param.num_samples)

    # ado ids
    ado_ids_array = collect(keys(prediction_dict));

    if length(ado_ids_array) == 0
        # if no pedestrian exists, return (num_samples*num_controls, prediction_steps + 1, 0, 2) array
        ap_array = Array{Float64, 4}(undef, sim_param.num_samples*num_controls, sim_param.prediction_steps + 1, 0, 2);
        return ap_array, time_idx_ap_array, control_idx_ex_array, ado_ids_array
    end

    ap_init_array = permutedims(cat([w_init.ap_dict[id] for id in ado_ids_array]..., dims=4),
                                    [2, 3, 4, 1]); # (1, 1, num_ado_agents, 2) array of initial ado positions
    ap_init_array = repeat(ap_init_array, inner=(sim_param.num_samples*num_controls, 1, 1, 1));

    ap_pred_array = permutedims(cat([prediction_dict[id] for id in ado_ids_array]..., dims=4),
                                      [1, 2, 4, 3]); # (num_samples*num_controls, prediction_steps, num_ado_agents, 2) array
    ap_array = cat(ap_init_array, ap_pred_array, dims=2) # (num_samples*num_controls, prediction_steps + 1, num_ado_agents, 2)

    return ap_array, time_idx_ap_array, control_idx_ex_array, ado_ids_array
end

# # Compute costs based on the forward simulation & predicted ado positions
function compute_costs(ex_array_gpu::CuArray{Float32, 3},
                       u_array_gpu::CuArray{Float32, 3},
                       ap_array_gpu::CuArray{Float32, 4},
                       time_idx_ap_array_gpu::CuArray{Int32, 1},
                       control_idx_ex_array_gpu::CuArray{Int32, 1},
                       target_pos_array_gpu::CuArray{Float32, 2},
                       cost_param::DRCCostParameter)
    if name(CuDevice(0)) == "NVIDIA GeForce RTX 3060"
        # Instataneous costs
        inst_cnt_cost_array_gpu = instant_control_cost(u_array_gpu, cost_param, threads=(16, 64));
        inst_pos_cost_array_gpu = instant_position_cost(ex_array_gpu, target_pos_array_gpu, cost_param, threads=(16, 64));
        inst_col_cost_array_gpu = instant_collision_cost(ex_array_gpu, ap_array_gpu,
                                                         time_idx_ap_array_gpu, control_idx_ex_array_gpu,
                                                         cost_param, threads=(64, 4, 4));
        # Terminal costs
        term_pos_cost_array_gpu = terminal_position_cost(ex_array_gpu, target_pos_array_gpu, cost_param, threads=(1024, 1));
        term_col_cost_array_gpu = terminal_collision_cost(ex_array_gpu, ap_array_gpu,
                                                          time_idx_ap_array_gpu, control_idx_ex_array_gpu,
                                                          cost_param, threads=(128, 1, 8));
    else
        # Instataneous costs
        inst_cnt_cost_array_gpu = instant_control_cost(u_array_gpu, cost_param);
        inst_pos_cost_array_gpu = instant_position_cost(ex_array_gpu, target_pos_array_gpu, cost_param);
        inst_col_cost_array_gpu = instant_collision_cost(ex_array_gpu, ap_array_gpu,
                                                         time_idx_ap_array_gpu, control_idx_ex_array_gpu,
                                                         cost_param);
        # Terminal costs
        term_pos_cost_array_gpu = terminal_position_cost(ex_array_gpu, target_pos_array_gpu, cost_param);
        term_col_cost_array_gpu = terminal_collision_cost(ex_array_gpu, ap_array_gpu,
                                                          time_idx_ap_array_gpu, control_idx_ex_array_gpu,
                                                          cost_param);
    end

    return SimulationCostResult(inst_cnt_cost_array_gpu, inst_pos_cost_array_gpu,
                                inst_col_cost_array_gpu, term_pos_cost_array_gpu,
                                term_col_cost_array_gpu)
end

# # Integrate all the costs over horizon, returning sampled total costs and the risk value
function integrate_costs(cost_result::SimulationCostResult,
                         sim_param::SimulationParameter,
                         constraint_time::Union{Nothing, Float64}=nothing)
    if !isnothing(constraint_time)
        c_idx = Int64(round(constraint_time/sim_param.dtc, digits=5))
        @assert c_idx < size(cost_result.inst_pos_cost_array_gpu, 2);
    else
        c_idx = size(cost_result.inst_cnt_cost_array_gpu, 2);
    end
    # control cost
    cnt_cost_per_control = sum(cost_result.inst_cnt_cost_array_gpu[:, 1:c_idx], dims=2).*Float32(sim_param.dtc);

    # position cost
    pos_cost_per_control = sum(cost_result.inst_pos_cost_array_gpu[:, 1:c_idx], dims=2).*Float32(sim_param.dtc);
    if isnothing(constraint_time)
        pos_cost_per_control += cost_result.term_pos_cost_array_gpu;
    end

    # for collision cost, sum over all timesteps and ado agents but not over samples
    col_cost_per_sample_control = sum(cost_result.inst_col_cost_array_gpu[:, 1:c_idx, :], dims=(2, 3)).*Float32(sim_param.dtc);
    if isnothing(constraint_time)
        col_cost_per_sample_control += sum(cost_result.term_col_cost_array_gpu, dims=3);
    end
    # reshape collision cost into (num_controls, num_samples) array
    col_cost_per_control_sample = permutedims(reshape(col_cost_per_sample_control, sim_param.num_samples, :),
                                              [2, 1]);
    @assert isa(col_cost_per_control_sample, CuArray{Float32, 2});
    num_controls = size(col_cost_per_control_sample, 1);

    # (num_controls, num_samples) array of Float64
    total_cost_per_control_sample = Float64.(collect(col_cost_per_control_sample .+ (cnt_cost_per_control + pos_cost_per_control)));

    # compute risk value per control
    risk_per_control = Vector{Float64}(undef, num_controls);
    for ii = 1:num_controls
        risk_value = sum(total_cost_per_control_sample[ii, :])./sim_param.num_samples
        @inbounds risk_per_control[ii] = risk_value
    end
    return total_cost_per_control_sample, risk_per_control
end

# Choose best nominal control (i.e. with lowest risk value) for backward simulation
function choose_best_nominal_control(total_cost_per_control_sample, risk_per_control, u_array_gpu)
    minimum_risk, best_control_idx = findmin(risk_per_control);
    sampled_total_costs = total_cost_per_control_sample[best_control_idx, :];

    best_u_array_tmp = Float64.(collect(u_array_gpu[best_control_idx, :, :]));
    best_u_array = [best_u_array_tmp[ii, :] for ii = 1:size(best_u_array_tmp, 1)];

    return sampled_total_costs, minimum_risk, best_control_idx, best_u_array
end

# # Compute cost gradients based on the forward simulation & predicted ado positions
function compute_cost_gradients(best_ex_array_gpu::CuArray{Float32, 2},
                                best_u_array_gpu::CuArray{Float32, 2},
                                best_ap_array_gpu::CuArray{Float32, 4},
                                time_idx_ap_array_gpu::CuArray{Int32, 1},
                                target_pos_array_gpu::CuArray{Float32, 2},
                                cost_param::DRCCostParameter)
    if name(CuDevice(0)) == "NVIDIA GeForce RTX 3060"
        # Instantaneous costs
        inst_pos_cost_grad_array_gpu = instant_position_cost_gradient(best_ex_array_gpu, target_pos_array_gpu,
                                                                      cost_param, threads=256);
        inst_col_cost_grad_array_gpu = instant_collision_cost_gradient(best_ex_array_gpu, best_ap_array_gpu,
                                                                       time_idx_ap_array_gpu, cost_param,
                                                                       threads=(64, 4, 4));

        # Terminal costs
        term_pos_cost_grad_array_gpu = terminal_position_cost_gradient(best_ex_array_gpu, target_pos_array_gpu,
                                                                       cost_param);
        term_col_cost_grad_array_gpu = terminal_collision_cost_gradient(best_ex_array_gpu, best_ap_array_gpu,
                                                                        time_idx_ap_array_gpu, cost_param, threads=(128, 1, 8));
    else
        # Instantaneous costs
        inst_pos_cost_grad_array_gpu = instant_position_cost_gradient(best_ex_array_gpu, target_pos_array_gpu,
                                                                      cost_param);
        inst_col_cost_grad_array_gpu = instant_collision_cost_gradient(best_ex_array_gpu, best_ap_array_gpu,
                                                                       time_idx_ap_array_gpu, cost_param);

        # Terminal costs
        term_pos_cost_grad_array_gpu = terminal_position_cost_gradient(best_ex_array_gpu, target_pos_array_gpu,
                                                                       cost_param);
        term_col_cost_grad_array_gpu = terminal_collision_cost_gradient(best_ex_array_gpu, best_ap_array_gpu,
                                                                        time_idx_ap_array_gpu, cost_param);
    end
    return SimulationCostGradResult(inst_pos_cost_grad_array_gpu, inst_col_cost_grad_array_gpu,
                                    term_pos_cost_grad_array_gpu, term_col_cost_grad_array_gpu)
end

# # sum cost gradients per timestep and sample, reshaping them to (4, total_timesteps, num_samples) array
function sum_cost_gradients(cost_grad_result::SimulationCostGradResult)
    num_samples = size(cost_grad_result.inst_col_cost_grad_array_gpu, 1);
    total_timesteps = size(cost_grad_result.inst_col_cost_grad_array_gpu, 2) + 1;

    cost_grad_array = Array{Float64, 3}(undef, 4, total_timesteps, num_samples);

    # sum over all ado agents
    inst_col_cost_grads_gpu = sum(cost_grad_result.inst_col_cost_grad_array_gpu, dims=3); # (num_samples, total_timesteps-1, 1, 4)
    term_col_cost_grads_gpu = sum(cost_grad_result.term_col_cost_grad_array_gpu, dims=3); # (num_samples, 1, 1, 4)

    cost_grad_array[:, 1:end-1, :] = collect(dropdims(permutedims(inst_col_cost_grads_gpu, [4, 3, 2, 1]), dims=2)); # (4, total_timesteps-1, num_samples)
    cost_grad_array[:, 1:end-1, :] .+= collect(permutedims(cost_grad_result.inst_pos_cost_grad_array_gpu, [2, 1])); # (4, total_timesteps-1, num_samples)
    cost_grad_array[:, end:end, :] = collect(dropdims(permutedims(term_col_cost_grads_gpu, [4, 3, 2, 1]), dims=2)); # (4, 1, num_samples)
    cost_grad_array[:, end:end, :] .+= collect(permutedims(cost_grad_result.term_pos_cost_grad_array_gpu, [2, 1])); # (4, 1, num_samples)

    return cost_grad_array
end

# # backward simulation of the costates.
function simulate_backward(best_ex_array::Array{Float64, 2}, # (total_timesteps, 4) array
                           best_u_array::Vector{Vector{Float64}},
                           cost_grad_array::Array{Float64, 3},
                           sim_param::SimulationParameter)
    e_costate_array = similar(cost_grad_array); # (4, total_timesteps, num_samples) array
    # boundary conditions
    e_costate_array[:, end, :] = cost_grad_array[:, end, :];

    for jj = Iterators.reverse(1:size(e_costate_array, 2) - 1)
        # backward euler integration
        @inbounds Jacobian =
            transition_jacobian(best_ex_array[jj + 1, :],
                                best_u_array[jj]) # Note the index jj here
        @inbounds costate_vel =
            -cost_grad_array[:, jj, :] .- # Note the index jj here
            transpose(Jacobian)*e_costate_array[:, jj + 1, :];
        @inbounds e_costate_array[:, jj, :] =
            e_costate_array[:, jj + 1, :] .+
            costate_vel*(-sim_param.dtc);
    end
    return e_costate_array
end

# Main Simulation Function (forward-backward simulation)
# Ado positions and predictions must be updated separately.
function simulate(w_init::WorldState,
                  u_arrays::Vector{Vector{Vector{Float64}}},
                  target_trajectory::Trajectory2D,
                  prediction_dict::Dict{String, Array{Float64, 3}},
                  sim_param::SimulationParameter,
                  constraint_time::Union{Nothing, Float64}=nothing)
    # process u_arrays
    u_array = process_u_arrays(u_arrays);
    u_array_gpu = cu(u_array);

    # ego state simulation
    ex_array_gpu = simulate_forward(w_init.e_state, u_array_gpu, sim_param);

    # get measurement schedule
    measurement_schedule = get_measurement_schedule(w_init, sim_param);

    # process prediction_dict for further computation
    # note that each value in prediction_dict has to be (num_samples*num_controls, prediction_steps, 2) array
    ap_array, time_idx_ap_array, control_idx_ex_array, ado_ids_array =
        process_ap_dict(ex_array_gpu, w_init, measurement_schedule, prediction_dict, sim_param);
    ap_array_gpu = cu(ap_array)
    time_idx_ap_array_gpu = CuArray{Int32}(time_idx_ap_array);
    control_idx_ex_array_gpu = CuArray{Int32}(control_idx_ex_array);

    # get target_pos array
    target_pos_array = get_target_pos_array(ex_array_gpu, w_init, target_trajectory, sim_param)
    target_pos_array_gpu = cu(target_pos_array);

    # compute costs
    cost_result = compute_costs(ex_array_gpu, u_array_gpu, ap_array_gpu,
                                time_idx_ap_array_gpu, control_idx_ex_array_gpu,
                                target_pos_array_gpu, sim_param.cost_param);

    # integrate costs
    total_cost_per_control_sample, risk_per_control = integrate_costs(cost_result, sim_param);

    # choose best nominal control
    sampled_total_costs, minimum_risk, best_control_idx, best_u_array =
        choose_best_nominal_control(total_cost_per_control_sample, risk_per_control, u_array_gpu);
    best_ex_array_gpu = ex_array_gpu[best_control_idx, :, :];
    best_ex_array = Float64.(collect(best_ex_array_gpu));
    best_u_array_gpu = u_array_gpu[best_control_idx, :, :];
    best_ap_array_sample_idx =
        sim_param.num_samples*(best_control_idx-1)+1:sim_param.num_samples*best_control_idx
    best_ap_array_gpu = ap_array_gpu[best_ap_array_sample_idx, :, :, :];
    # compute cost gradients using the best nominal control
    cost_grad_result = compute_cost_gradients(best_ex_array_gpu, best_u_array_gpu,
                                              best_ap_array_gpu, time_idx_ap_array_gpu,
                                              target_pos_array_gpu,
                                              sim_param.cost_param);

    # get cost gradient array for costate computation
    cost_grad_array = sum_cost_gradients(cost_grad_result);

    # backward simulation of ego costate
    e_costate_array = simulate_backward(best_ex_array, best_u_array, cost_grad_array, sim_param);

    if !isnothing(constraint_time)
        # compute indices to slice best_ex_array, best_u_array, cost_grad_array for constraint enforcement
        c_idx = Int64(round(constraint_time/sim_param.dtc, digits=5)) + 1;
        @assert c_idx < size(best_ex_array, 1)
        # compute adjoint for constraint enforcement
        cost_grad_array[:, c_idx, :] = zeros(size(cost_grad_array[:, c_idx, :])); # TODO: maybe we do not want to mutate cost_grad_array.
        e_costate_array_constraint = simulate_backward(best_ex_array[1:c_idx, :],
                                                       best_u_array[1:c_idx],
                                                       cost_grad_array[:, 1:c_idx, :],
                                                       sim_param);
        total_cost_per_control_sample_constraint, risk_per_control_constraint =
            integrate_costs(cost_result, sim_param, constraint_time);
        sampled_total_costs_constraint =
            total_cost_per_control_sample_constraint[best_control_idx, :];
        risk_constraint = risk_per_control_constraint[best_control_idx];
    else
        e_costate_array_constraint = nothing;
        sampled_total_costs_constraint = nothing;
        risk_constraint = nothing;
    end

    sim_result = SimulationResult(best_u_array,
                                  best_control_idx,
                                  simulate_forward(w_init.e_state, best_u_array, sim_param),
                                  measurement_schedule, prediction_dict,
                                  best_ap_array_gpu,
                                  ado_ids_array, sampled_total_costs, minimum_risk,
                                  e_costate_array,
                                  e_costate_array_constraint,
                                  sampled_total_costs_constraint,
                                  risk_constraint)
    return sim_result, best_u_array
end

#=
# Forward (re-)simulation and evaluation of risk (for potential line search which is not implemented yet)
function evaluate_risk(w_init::WorldState,
                       u_array::Vector{Vector{Float64}},
                       target_trajectory::Trajectory2D,
                       ap_array_gpu::CuArray{Float32, 4},
                       sim_param::SimulationParameter)
    # ego state simulation
    e_state_array = simulate_forward(w_init.e_state, u_array, sim_param);
    # export to gpu
    ep_array_gpu = cu(hcat(get_position.(e_state_array)...));
    # compute costs
    cost_result = compute_costs(e_state_array, ep_array_gpu, u_array, ap_array_gpu,
                                target_trajectory, sim_param.cost_param);
    ~, risk_val = integrate_costs(cost_result, sim_param);
    return risk_val
end
=#

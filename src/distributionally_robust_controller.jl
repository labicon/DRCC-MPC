#///////////////////////////////////////
#// File Name: distributionally_robust_controller.jl
#// Author: Kanghyun Ryu (kr37@illinois.edu)
#// Date Created: 2023/04/01
#// Description: Distributionally Robust Controller
#///////////////////////////////////////

using DataStructures
using LinearAlgebra
using Printf
using Random
using RobotOS
import Convex: Variable, norm, quadform, minimize, dot, solve!
using ECOS
using Statistics

struct DRCControlParameter <: Parameter
    eamax::Float64  # Maximum abolute value of acceleration
    tcalc::Float64  # Allocated control computation time
    goal_pos::Vector{Float64} # [x, y] goal position
    dtr::Float64 # Replanning time interval
    dtc::Float64 # Euler integration time interval

    horizon::Int64 # Planning horizon

    human_size::Float64 # Human size

    cem_init_mean::Vector{Float64}
    cem_init_cov::Matrix{Float64}
    cem_init_num_samples::Int64
    cem_init_num_elites::Int64
    cem_init_alpha::Float64
    cem_init_iterations::Int64

    epsilon::Float64 # Risk-sensitiveness parameter

    function DRCControlParameter(eamax::Float64, tcalc::Float64, 
                                goal_pos::Vector{Float64}, dtr::Float64, dtc::Float64,
                                horizon::Int64,
                                human_size::Float64,
                                cem_init_mean::Vector{Float64},
                                cem_init_cov::Matrix{Float64},
                                cem_init_num_samples::Int64,
                                cem_init_num_elites::Int64,
                                cem_init_alpha::Float64,
                                cem_init_iterations::Int64,
                                epsilon::Float64)
        @assert eamax > 0.0 "eamax must be positive."
        @assert tcalc > 0.0 "tcalc must be positive."
        @assert dtr > 0.0 "dtr must be positive."
        @assert length(goal_pos) == 2 "goal_pos must be a 2D vector."
        return new(eamax, tcalc, goal_pos, dtr, dtc, horizon, human_size,
                    cem_init_mean, cem_init_cov,
                    cem_init_num_samples, cem_init_num_elites, cem_init_alpha,
                    cem_init_iterations, epsilon)
    end
end

mutable struct DRCController
    sim_param::SimulationParameter
    cnt_param::DRCControlParameter
    predictor::Predictor
    cost_param::DRCCostParameter

    prediction_dict::Union{Nothing, Dict{String, Array{Float64, 3}}}
    sim_result::Union{Nothing, SimulationResult}
    tcalc_actual::Union{Nothing, Float64}

    prediction_task::Union{Nothing, Task}
    control_update_task::Union{Nothing, Task}

    prediction_dict_tmp::Union{Nothing, Dict{String, Array{Float64, 3}}}
    sim_result_tmp::Union{Nothing, SimulationResult}
    tcalc_actual_tmp::Union{Nothing, Float64}
    u_value_tmp::Union{Nothing, Vector{Float64}}
    u_value::Union{Nothing, Vector{Float64}}
end

function DRCController(sim_param::SimulationParameter,
                        cnt_param::DRCControlParameter,
                        predictor::Predictor,
                        cost_param::DRCCostParameter)
    return DRCController(sim_param, cnt_param, predictor, cost_param, nothing, nothing, nothing,
                        nothing, nothing, nothing, nothing, nothing, nothing, nothing)
end

# main control functions below
function control!(controller::DRCController,
                    current_time::Time,
                    log::Union{Nothing, Vector{Tuple{Time, String}}}=nothing);
    if !isnothing(controller.prediction_task) &&
        istaskdone(controller.prediction_task)
        if !isnothing(log)
            msg = "New prediction is available to the controller."
            push!(log, (current_time, msg))
        end
        controller.prediction_dict = copy(controller.prediction_dict_tmp);
        controller.prediction_task = nothing;
    end

    if !isnothing(controller.control_update_task) &&
        istaskdone(controller.control_update_task)
        if !isnothing(log)
            msg = "New Distributionally Robust control is available to the controller"
            push!(log, (current_time, msg))
        end
        controller.tcalc_actual = copy(controller.tcalc_actual_tmp);
        controller.u_value = copy(controller.u_value_tmp);
        controller.control_update_task = nothing;
    end
    u = copy(controller.u_value);
    if !isnothing(log)
        msg = "control: $(u) is applied to the system."
        push!(log, (current_time, msg))
    end
    return u
end

function schedule_prediction!(controller::DRCController,
                                ado_pos_dict::Dict,
                                e_init::Union{Nothing, RobotState}=nothing);
    if typeof(controller.predictor) == TrajectronPredictor
        controller.prediction_task = @task begin
            if controller.predictor.param.use_robot_future
                @assert !isnothing(e_init) "e_init must be given."
                robot_present_and_future=
                    get_robot_present_and_future(e_init,
                                                controller.u_schedule,
                                                controller.sim_param,
                                                controller.cnt_param);
            else
                robot_present_and_future = nothing;
            end
            controller.prediction_dict_tmp = 
                sample_future_ado_positions!(controller.predictor,
                                                ado_pos_dict,
                                                robot_present_and_future);
            if !controller.predictor.param.use_robot_future
                num_controls = 1
                # each value has to be (num_samples*num_controls, prediction_steps, 2) array
                for key in keys(controller.prediction_dict_tmp)
                    controller.prediction_dict_tmp[key] = 
                        repeat(controller.prediction_dict_tmp[key],
                                outer=(num_controls, 1, 1));
                end
            end
        end
    else
        @error "Type of controller.predictor: $(typeof(controller.predictor)) is not supported."
    end
    schedule(controller.prediction_task)
end

function schedule_control_update!(controller::DRCController,
                                    w_init::WorldState,
                                    log::Union{Nothing, Vector{Tuple{Time, String}}}=nothing);
    if !isnothing(controller.prediction_task) &&
        istaskdone(controller.prediction_task)
        if !isnothing(log)
            msg = "New prediction is available to the controller."
            push!(log, (w_init.t, msg))
        end
        controller.prediction_dict = copy(controller.prediction_dict_tmp);
        controller.prediction_task = nothing;
    end
    if !isnothing(log)
        msg = "New Distributionally Robust control is scheduled."
        push!(log, (w_init.t, msg))
    end

    controller.control_update_task = @task begin
        controller.tcalc_actual_tmp, controller.u_value_tmp = 
            drc_control_update!(controller, controller.cnt_param,
                        controller.prediction_dict,
                        w_init);
    end
    schedule(controller.control_update_task)
end

# helper functions below
@inline function drc_control_update!(controller::DRCController,
                                    cnt_param::DRCControlParameter,
                                    prediction_dict::Dict{String, Array{Float64, 3}},
                                    w_init::WorldState)
    tcalc_actual = 
        @elapsed u = get_action!(controller, cnt_param, prediction_dict, w_init);

    if tcalc_actual >= cnt_param.tcalc
        # tcalc actual has exceeded allowable computation time
        time = @sprintf "Time %.2f" round(to_sec(w_init.t), digits=5)
        @warn "$(time) [sec]: DRC computation took $(round(tcalc_actual, digits=3)) [sec], which exceeds the maximum computation time allowed."
    end

    return tcalc_actual, u
end

function get_action!(controller::DRCController,
                    cnt_param::DRCControlParameter,
                    prediction_dict::Dict{String, Array{Float64, 3}},
                    w_init::WorldState)

    # compute mean & cov of human transitions
    mean_dict, cov_dict = get_mean_cov(prediction_dict);

    # CEM optimization 
    u = cem_optimization!(controller, cnt_param, mean_dict, cov_dict, w_init);
    return u
end

@inline function cem_optimization!(controller::DRCController,
                                    cnt_param::DRCControlParameter,
                                    prediction_mean_dict::Dict{String, Array{Float64, 2}},
                                    prediction_cov_dict::Dict{String, Array{Float64, 3}},
                                    w_init::WorldState)
    # initialize control candidates
    dist_mean = [cnt_param.cem_init_mean for i in 1:cnt_param.horizon];
    dist_cov = [cnt_param.cem_init_cov for i in 1:cnt_param.horizon];
    
    for iteration in 1:cnt_param.cem_init_iterations
        # sample control candidates
        u_candidates = zeros(cnt_param.cem_init_num_samples, cnt_param.horizon, 2);
        for i in 1:cnt_param.horizon
            if isposdef(dist_cov[i])
                u_candidates[:, i, :] = transpose(rand(MvNormal(dist_mean[i], dist_cov[i]), cnt_param.cem_init_num_samples));
            else
                dist_cov[i] = [1 0; 0 1];
                u_candidates[:, i, :] = transpose(rand(MvNormal(dist_mean[i], dist_cov[i]), cnt_param.cem_init_num_samples));
            end
        end
        clamp!(u_candidates, -cnt_param.eamax, cnt_param.eamax)
        # compute cost and CVaR for each control candidates
        cost, CVaR = compute_cost_CVaR(u_candidates, cnt_param, controller.sim_param, 
                                        prediction_mean_dict, prediction_cov_dict, w_init, controller.cost_param);
        
        # remove samples which violates CVaR constraint
        if all(CVaR .> 0.0)
            # if all samples violate CVaR constraints, then use the sample with the lowest CVaR
            order = sortperm(CVaR);
            u_candidates = u_candidates[order,:,:];
            if iteration == cnt_param.cem_init_iterations
                # if this is the last iteration, then use the sample with the lowest CVaR
                @warn "All samples violate CVaR constraints."
            end
        else
            # if some samples violate CVaR constraints, then remove them
            u_candidates = u_candidates[CVaR .< 0.0,:,:];
            cost = cost[CVaR .< 0.0];
            # sort cost and find elite control candidates
            order = sortperm(cost);
            cost = cost[order];
            u_candidates = u_candidates[order,:,:];
        end
        N_elite = min(cnt_param.cem_init_num_elites, size(u_candidates, 1));
        elite_samples = u_candidates[1:N_elite,:,:];
        
        # update mean and var
        new_mean = [];
        new_cov = [];
        for i in 1:cnt_param.horizon
            new_mean = vec(mean(elite_samples[:,i,:], dims=1));
            new_cov = cov(elite_samples[:,i,:], dims=1);
            # dist_mean[i] = cnt_param.cem_init_alpha*dist_mean[i] + (1-cnt_param.cem_init_alpha)*new_mean;
            # dist_cov[i] = cnt_param.cem_init_alpha*dist_cov[i] + (1-cnt_param.cem_init_alpha)*new_cov;
            dist_mean[i] = new_mean;
            dist_cov[i] = new_cov;
        end
    end
    u_optim = dist_mean[1];
    clamp!(u_optim, -cnt_param.eamax, cnt_param.eamax)

    return u_optim
end

# get mean and cov from prediction_dict
@inline function get_mean_cov(prediction_dict::Dict{String, Array{Float64, 3}})

    mean_dict = Dict{String, Array{Float64, 2}}();
    cov_dict = Dict{String, Array{Float64, 3}}();
    for key in keys(prediction_dict)
        # Each trajectory is a (num_samples, num_predicted_timesteps, 2) array
        traj = prediction_dict[key];
        # get mean
        mean_dict[key] = dropdims(mean(traj, dims=1), dims=1);
        # get covariance
        covariance = zeros(size(traj, 2), size(traj, 3), size(traj, 3));
        for time_step in axes(traj, 2)
            covariance[time_step, :, :] = cov(traj[:, time_step, :]);
        end
        cov_dict[key] = covariance;
    end

    return mean_dict, cov_dict
end

@inline function compute_cost_CVaR(u_candidates::Array{Float64, 3},
                                cnt_param::DRCControlParameter,
                                sim_param::SimulationParameter,
                                prediction_mean_dict::Dict{String, Array{Float64, 2}},
                                prediction_cov_dict::Dict{String, Array{Float64, 3}},
                                w_init::WorldState,
                                cost_param::DRCCostParameter)
    # compute cost and CVaR for each control candidates
    cost = zeros(size(u_candidates, 1));
    CVaR = zeros(size(u_candidates, 1));

    # ratio between dto and dtc
    euler_expansion_factor = Int64(cnt_param.dtr/cnt_param.dtc);
    sim_expansion_factor = Int64(sim_param.dto/cnt_param.dtc);
    cnt_idx = Vector(1:cnt_param.horizon)*euler_expansion_factor;
    predict_idx = Vector(1:sim_param.prediction_steps)*sim_expansion_factor;

    for i in 1:size(u_candidates, 1)
        u = Vector{Vector{Float64}}(undef, cnt_param.horizon*euler_expansion_factor);
        for j in 1:cnt_param.horizon
            for k in 1:euler_expansion_factor
                u[(j-1)*euler_expansion_factor+k] = u_candidates[i,j,:];
            end
        end
        # forward simulation of inputs
        sim_result = simulate_forward(w_init.e_state, u, sim_param);
        # compute cost
        cost[i] = compute_cost(sim_result[2:end], u, cost_param, cnt_idx);
        # compute CVaR
        CVaR[i] = compute_CVaR(sim_result[2:end], cnt_param, prediction_mean_dict, prediction_cov_dict, predict_idx);
    end

    return cost, CVaR
end

@inline function compute_cost(sim_results::Vector{RobotState},
                        u::Vector{Vector{Float64}},
                        cost_param::DRCCostParameter,
                        cnt_idx::Vector{Int64});
    cost = 0.0;

    for i in cnt_idx
        cost += instant_position_cost(sim_results[i], cost_param);
        if i < length(sim_results)
            cost += instant_control_cost(u[i], cost_param);
        end
    end
    return cost
end

@inline function compute_CVaR(sim_result::Vector{RobotState},
                            cnt_param::DRCControlParameter,
                            prediction_mean_dict::Dict{String, Array{Float64, 2}},
                            prediction_cov_dict::Dict{String, Array{Float64, 3}},
                            predict_idx::Vector{Int64});
    CVaR = [];

    # radius of human 
    r = cnt_param.human_size;
    epsilon = cnt_param.epsilon;

    for key in keys(prediction_mean_dict)
        # get mean and cov
        mean = prediction_mean_dict[key];
        cov = prediction_cov_dict[key];
        for (pred_idx, euler_idx) in enumerate(predict_idx)
            e_position = get_position(sim_result[euler_idx]);
            # compute distance between mean and ego agent
            dist = norm(mean[pred_idx, :] - e_position);
            # compute CVaR
            append!(CVaR, -(dist - r)^2 + 1/epsilon * tr(cov[pred_idx, :, :]));
        end
    end

    if isempty(CVaR)
        return -1.0
    else
        return maximum(CVaR)
    end
end

# for Trajectron robot-future-conditoinal models
@inline function get_robot_present_and_future(e_init::RobotState,
                                                u_schedule::OrderedDict{Time, Vector{Float64}},
                                                sim_param::SimulationParameter,
                                                cnt_param::DRCControlParameter)
    # compute u_arrays from u_schedule and nominal control candidates
    u_arrays = get_nominal_u_arrays(u_schedule, sim_param, cnt_param);
    # process u_arrays
    u_array_gpu = cu(process_u_arrays(u_arrays));
    # ego state simulation
    ex_array_gpu = simulate_forward(e_init, u_array_gpu, sim_param)
    slice_factor = Int64(round(sim_param.dto/sim_param.dtc, digits=5));
    ex_array_cpu = collect(ex_array_gpu[:, 1:slice_factor:end, :])
    @assert size(ex_array_cpu, 2) == sim_param.prediction_steps + 1

    # (num_controls, 1 + prediction_steps, 6) array where the last dimension
    # is [pos_x, pos_y, vel_x, vel_y, acc_x, acc_y]. accelerations are computed
    # by finite-differencing velocity snapshots.
    robot_present_and_future = Array{Float64, 3}(undef, size(ex_array_cpu, 1),
        size(ex_array_cpu, 2),
        6);
    robot_present_and_future[:, :, 1:4] = ex_array_cpu;
    acc = diff(ex_array_cpu[:, :, 3:4], dims=2)./sim_param.dto
    robot_present_and_future[:, 1:end-1, 5:6] = acc;
    # pad last acceleration
    robot_present_and_future[:, end, 5:6] = robot_present_and_future[:, end-1, 5:6]

    return robot_present_and_future
end
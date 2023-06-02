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
    discount::Float64 # Discount factor

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
                                discount::Float64,
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
        return new(eamax, tcalc, goal_pos, dtr, dtc, horizon, discount, human_size,
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

    previous_cnt_plan::Union{Nothing, Array{Float64, 2}}
end

function DRCController(sim_param::SimulationParameter,
                        cnt_param::DRCControlParameter,
                        predictor::Predictor,
                        cost_param::DRCCostParameter)
    return DRCController(sim_param, cnt_param, predictor, cost_param, nothing, nothing, nothing,
                        nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
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

function adjust_old_prediction!(controller::DRCController,
                                previous_ado_pos_dict::Dict{String, Vector{Float64}},
                                latest_ado_pos_dict::Dict{String, Vector{Float64}})
    @assert keys(controller.prediction_dict) == keys(previous_ado_pos_dict)

    ado_agents_removed = setdiff(keys(previous_ado_pos_dict),
                                    keys(latest_ado_pos_dict));
    ado_agents_added = setdiff(keys(latest_ado_pos_dict),
                                keys(previous_ado_pos_dict));
    keys_list = collect(keys(controller.prediction_dict))
    for key in keys_list
        if in(key, ado_agents_removed)
            # If this ado_agent is removed in latest_ado_pos_dict, then remove
            # it from prediction_dict.
            pop!(controller.prediction_dict, key)
        else
            if typeof(controller.predictor) != OraclePredictor # skip this step for OraclePredictor as predictions are perfect.
                # If this ado_agent is existent in both previous and latest
                # ado_pos_dict, then reuse and adjust the previous prediction.
                (diff_x, diff_y) = latest_ado_pos_dict[key] -
                                    previous_ado_pos_dict[key];
                controller.prediction_dict[key][:, :, 1] .+= diff_x;
                controller.prediction_dict[key][:, :, 2] .+= diff_y;
            end
        end
    end
    for key in ado_agents_added
        # Treat this newly added ado_agent as a static obstacle
        # (until the new prediction becomes available to the controller)
        num_controls = 1;
        controller.prediction_dict[key] =
            zeros(controller.sim_param.num_samples*num_controls,
                    controller.sim_param.prediction_steps, 2);
        controller.prediction_dict[key][:, :, 1] .+= latest_ado_pos_dict[key][1];
        controller.prediction_dict[key][:, :, 2] .+= latest_ado_pos_dict[key][2];
    end
    @assert keys(controller.prediction_dict) == keys(latest_ado_pos_dict)
end

function schedule_prediction!(controller::DRCController,
                                ado_pos_dict::Dict,
                                previous_ado_pos_dict::Union{Nothing, Dict{String, Vector{Float64}}}=nothing,
                                e_init::Union{Nothing, RobotState}=nothing);
    if !isnothing(previous_ado_pos_dict)
        adjust_old_prediction!(controller, previous_ado_pos_dict,
                                convert_nodes_to_str(reduce_to_positions(ado_pos_dict)));
    end
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
    elseif typeof(controller.predictor) == GaussianPredictor || typeof(controller.predictor) == StopGaussianPredictor
        controller.prediction_task = @task begin
            controller.prediction_dict_tmp =
                sample_future_ado_positions!(controller.predictor,
                                             convert_nodes_to_str(ado_pos_dict));
            num_controls = 1;
            # each value has to be (num_samples*num_controls, prediction_steps, 2) array
            for key in keys(controller.prediction_dict_tmp)
                controller.prediction_dict_tmp[key] =
                    repeat(controller.prediction_dict_tmp[key],
                           outer=(num_controls, 1, 1));
            end
        end
    else
        @error "Type of controller.predictor: $(typeof(controller.predictor)) is not supported."
    end
    schedule(controller.prediction_task)
end

function schedule_control_update!(controller::DRCController,
                                    w_init::WorldState,
                                    target_trajectory::Trajectory2D;
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
                        w_init, target_trajectory);
    end
    schedule(controller.control_update_task)
end

# helper functions below
function drc_control_update!(controller::DRCController,
                                    cnt_param::DRCControlParameter,
                                    prediction_dict::Dict{String, Array{Float64, 3}},
                                    w_init::WorldState,
                                    target_trajectory::Trajectory2D)
    tcalc_actual = 
        @elapsed u = get_action!(controller, cnt_param, prediction_dict, w_init, target_trajectory);

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
                    w_init::WorldState,
                    target_trajectory::Trajectory2D)

    # compute mean & cov of human transitions
    mean_dict, cov_dict = get_mean_cov(prediction_dict);

    # CEM optimization 
    u = cem_optimization!(controller, cnt_param, mean_dict, cov_dict, w_init, target_trajectory);
    return u
end

function cem_optimization!(controller::DRCController,
                                    cnt_param::DRCControlParameter,
                                    prediction_mean_dict::Dict{String, Array{Float64, 2}},
                                    prediction_cov_dict::Dict{String, Array{Float64, 3}},
                                    w_init::WorldState,
                                    target_trajectory::Trajectory2D)
    # initialize control candidates
    # if isnothing(controller.previous_cnt_plan)
    #     dist_mean = [cnt_param.cem_init_mean for i in 1:cnt_param.horizon];
    #     dist_var = [[1.0, 1.0] .* cnt_param.eamax^2 for i in 1:cnt_param.horizon];
    # else
    #     dist_mean = [vec(controller.previous_cnt_plan[i,:]) for i in 2:cnt_param.horizon];
    #     push!(dist_mean, vec(controller.previous_cnt_plan[end,:]));
    #     dist_var = [[1.0, 1.0] .* cnt_param.eamax^2 for i in 1:cnt_param.horizon];
    # end
    dist_mean = ones(cnt_param.horizon, 2) .* cnt_param.cem_init_mean';
    dist_var = ones(cnt_param.horizon, 2) .* cnt_param.eamax^2;

    for iteration in 1:cnt_param.cem_init_iterations
        # sample control candidates
        u_candidates = zeros(cnt_param.cem_init_num_samples, cnt_param.horizon, 2);

        for i in 1:cnt_param.horizon
            lb_dist = dist_mean[i,:] .+ cnt_param.eamax;
            ub_dist = cnt_param.eamax .- dist_mean[i,:];
            dist_var[i,:] = min(min((lb_dist/2).^2, ((ub_dist/2).^2)), dist_var[i,:]);
            u_candidates[:, i, :] = sqrt.(dist_var[i,:])' .* rand(Normal(0.0, 1.0), (cnt_param.cem_init_num_samples, 2)) .+ dist_mean[i,:]';
        end
        clamp!(u_candidates, -cnt_param.eamax, cnt_param.eamax)
        # compute cost and CVaR for each control candidates
        cost, CVaR_sum, CVaR_max = compute_cost_CVaR(u_candidates, cnt_param, controller.sim_param, target_trajectory,
                                        prediction_mean_dict, prediction_cov_dict, w_init, controller.cost_param);
        # remove samples which violates CVaR constraint
        if all(CVaR_max .>= 0.0)
            # if all samples violate CVaR constraints, then use the sample with the lowest CVaR
            order = sortperm(CVaR_sum);
            u_candidates = u_candidates[order,:,:];
            if iteration == cnt_param.cem_init_iterations
                # if this is the last iteration, then use the sample with the lowest CVaR
                t = @sprintf "Time %.2f" round(to_sec(w_init.t), digits=5)
                @warn "$(t) [sec]: All samples violate CVaR constraints."
            end
        else
            # if some samples violate CVaR constraints, then remove them
            u_candidates = u_candidates[CVaR_max .< 0.0,:,:];
            cost = cost[CVaR_max .< 0.0];
            # sort cost and find elite control candidates
            order = sortperm(cost);
            cost = cost[order];
            u_candidates = u_candidates[order,:,:];
        end

        N_elite = min(cnt_param.cem_init_num_elites, size(u_candidates, 1));
        elite_samples = u_candidates[1:N_elite,:,:];
        
        # update mean and var
        new_mean = dropdims(mean(elite_samples, dims=1), dims=1);
        new_var = dropdims(var(elite_samples, dims=1), dims=1);
        dist_mean = cnt_param.cem_init_alpha * dist_mean .+ (1-cnt_param.cem_init_alpha) * new_mean;
        dist_var = cnt_param.cem_init_alpha * dist_var .+ (1-cnt_param.cem_init_alpha) * new_var;
        clamp!(dist_mean, -cnt_param.eamax, cnt_param.eamax)
        # for i in 1:cnt_param.horizon
        #     new_mean = vec(mean(elite_samples[:,i,:], dims=1));
        #     new_var = vec(var(elite_samples[:,i,:], dims=1));
        #     dist_mean[i] = cnt_param.cem_init_alpha*dist_mean[i] + (1-cnt_param.cem_init_alpha)*new_mean;
        #     dist_var[i] = cnt_param.cem_init_alpha*dist_var[i] + (1-cnt_param.cem_init_alpha)*new_var;
        #     clamp!(dist_mean[i], -cnt_param.eamax, cnt_param.eamax)
        # end
        controller.previous_cnt_plan = elite_samples[1,:,:];
        if norm(dist_var) < 1e-3 * cnt_param.horizon
            println("CEM optimization converged at iteration $(iteration)");
            print(norm(dist_var));
            break
        end
    end

    return controller.previous_cnt_plan[1,:]
end

# get mean and cov from prediction_dict
function get_mean_cov(prediction_dict::Dict{String, Array{Float64, 3}})

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

function compute_cost_CVaR(u_candidates::Array{Float64, 3},
                                cnt_param::DRCControlParameter,
                                sim_param::SimulationParameter,
                                target_trajectory::Trajectory2D,
                                prediction_mean_dict::Dict{String, Array{Float64, 2}},
                                prediction_cov_dict::Dict{String, Array{Float64, 3}},
                                w_init::WorldState,
                                cost_param::DRCCostParameter)
    # compute cost and CVaR for each control candidates
    cost = zeros(size(u_candidates, 1));
    CVaR_sum = zeros(size(u_candidates, 1));
    CVaR_max = zeros(size(u_candidates, 1));

    # # ratio between dto and dtc
    # pred_expansion_factor = Int64(sim_param.dto/cnt_param.dtc);
    # cnt_idx = Vector(1:cnt_param.horizon);
    # predict_idx = repeat(Vector(1:sim_param.prediction_steps), inner=pred_expansion_factor);

    # for i in 1:size(u_candidates, 1)
    #     u = Vector{Vector{Float64}}(undef, cnt_param.horizon);
    #     for j in 1:cnt_param.horizon
    #         u[j] = u_candidates[i,j,:];
    #     end
    #     # forward simulation of inputs
    #     sim_result = simulate_forward(w_init.e_state, u, sim_param);
    #     # compute cost
    #     cost[i] = compute_cost(sim_result[2:end], u, cost_param, cnt_param, cnt_idx, target_trajectory);
    #     # compute CVaR
    #     CVaR_sum[i], CVaR_max[i] = compute_CVaR(sim_result[2:end], w_init, cnt_param, prediction_mean_dict, prediction_cov_dict, predict_idx, pred_expansion_factor);
    # end

    # ratio between dto and dtc
    pred_expansion_factor = Int64(sim_param.dto/cnt_param.dtc);
    cnt_idx = Vector(1:cnt_param.horizon);
    predict_idx = repeat(Vector(1:sim_param.prediction_steps), inner=pred_expansion_factor);

    # Process u_arrays
    u_array_gpu = cu(u_candidates);

    # ego state simulation
    ex_array_gpu = simulate_forward(w_init.e_state, u_array_gpu, sim_param) # (num_candidate, total_timesteps, 4(pos + vel));
    ex_array_cpu = collect(ex_array_gpu);

    # get target_pos array
    total_timestep = size(ex_array_gpu, 2);
    target_pos_array = ones(total_timestep, 2) .* cost_param.ep_target'
    # target_pos_array = get_target_pos_array(ex_array_gpu, w_init, target_trajectory, sim_param);
    target_pos_array_gpu = cu(target_pos_array);

    # compute cost
    cost_result = compute_costs(ex_array_gpu, u_array_gpu, target_pos_array_gpu, cost_param);
    cost = integrate_costs(cost_result, sim_param);

    # for i in 1:size(u_candidates, 1)
    #     # compute CVaR
    #     CVaR_sum[i], CVaR_max[i] = compute_CVaR_array(ex_array_cpu[i, 2:end, :], w_init, cnt_param, prediction_mean_dict, prediction_cov_dict, predict_idx, pred_expansion_factor);
    # end
    CVaR_sum, CVaR_max = compute_CVaR_array_gpu(ex_array_cpu[:, 2:end, :], w_init, cnt_param, prediction_mean_dict, prediction_cov_dict, predict_idx, pred_expansion_factor);


    return cost, CVaR_sum, CVaR_max
end

function compute_cost(sim_results::Vector{RobotState},
                        u::Vector{Vector{Float64}},
                        cost_param::DRCCostParameter,
                        cnt_param::DRCControlParameter,
                        cnt_idx::Vector{Int64},
                        target_trajectory::Trajectory2D);
    cost = 0.0;

    for (horizon, i) in enumerate(cnt_idx)
        cost += cnt_param.discount^horizon * instant_position_cost(sim_results[i], cost_param);
        if i < length(sim_results)
            cost += cnt_param.discount^horizon * instant_control_cost(u[i], cost_param);
        end
    end
    return cost
end

function compute_CVaR_array(sim_result::Array{Float32, 2},
                            w_init::WorldState,
                            cnt_param::DRCControlParameter,
                            prediction_mean_dict::Dict{String, Array{Float64, 2}},
                            prediction_cov_dict::Dict{String, Array{Float64, 3}},
                            predict_idx::Vector{Int64},
                            pred_expansion_factor::Int64);

    CVaR = -1.0 .* ones(size(sim_result, 1));

    current_ado_position_dict = w_init.ap_dict;

    for key in keys(prediction_mean_dict)
        # get mean and cov
        current_pos = current_ado_position_dict[key]';
        mean = prediction_mean_dict[key];
        pos = vcat(current_pos, mean);
        interpolated_pos = Array{Float64, 2}(undef, size(sim_result, 1), 2);
        for i in 1:size(sim_result, 1)-1
            interpolate = (rem(i, pred_expansion_factor)/pred_expansion_factor)*pos[div(i, pred_expansion_factor)+2,:] + 
                    (1-(rem(i, pred_expansion_factor)/pred_expansion_factor))*pos[div(i, pred_expansion_factor)+1,:];
            interpolated_pos[i, :] = interpolate;
        end
        interpolated_pos[end, :] = pos[end, :];

        cov = prediction_cov_dict[key];
        for (euler_idx, pred_idx) in enumerate(predict_idx)
            e_position = sim_result[euler_idx, 1:2];
            # relative vector to the robot position from the human position
            rel_vec = e_position - interpolated_pos[euler_idx, :];
            # compute distance between mean and ego agent
            dist = norm(rel_vec) - cnt_param.human_size;
            if dist > 0.0
                # Find the ellipsoid
                # (x - p_human)^T E (x - p_human) = 1 & E = Q D Q^t
                R = maximum([100.0, dist]);
                D = diagm([1/dist^2, 1/R^2]);
                Q = [rel_vec[1]/dist rel_vec[2]/dist; -rel_vec[2]/dist rel_vec[1]/dist];
                E = Q*D*transpose(Q);
                # compute CVaR
                CVaR[euler_idx] = max(CVaR[euler_idx], -1 + 1/cnt_param.epsilon * tr(cov[pred_idx, :, :] * E));
                # append!(CVaR, -1 + 1/cnt_param.epsilon * tr(cov[pred_idx, :, :] * E));
            else
                CVaR[euler_idx] = 1.0;
            end
        end
    end

    for idx in 1:size(sim_result, 1)
        CVaR[idx] = CVaR[idx] * 0.9^(idx-1);
    end
    # if isempty(CVaR)
    #     return -100.0
    # else
    #     return maximum(CVaR)
    # end
    return sum(CVaR), maximum(CVaR)
end

function compute_CVaR_array_gpu(sim_result::Array{Float32, 2},
                            w_init::WorldState,
                            cnt_param::DRCControlParameter,
                            prediction_mean_dict::Dict{String, Array{Float64, 2}},
                            prediction_cov_dict::Dict{String, Array{Float64, 3}},
                            predict_idx::Vector{Int64},
                            pred_expansion_factor::Int64
                            threads::NTuple{2, Int}=(8, 32));

    CVaR = -1.0 .* ones(size(sim_result, 1), size(sim_result, 2));

    current_ado_position_dict = w_init.ap_dict;

    for key in keys(prediction_mean_dict)
        # get mean and cov
        current_pos = current_ado_position_dict[key]';
        mean = prediction_mean_dict[key];
        pos = vcat(current_pos, mean);
        interpolated_pos = Array{Float64, 2}(undef, size(sim_result, 1), 2);
        for i in 1:size(sim_result, 1)-1
            interpolate = (rem(i, pred_expansion_factor)/pred_expansion_factor)*pos[div(i, pred_expansion_factor)+2,:] + 
            (1-(rem(i, pred_expansion_factor)/pred_expansion_factor))*pos[div(i, pred_expansion_factor)+1,:];
            interpolated_pos[i, :] = interpolate;
        end
        interpolated_pos[end, :] = pos[end, :];

        interpolated_cov = repeat(prediction_cov_dict[key], inner = (pred_expansion_factor, 1, 1));

        pos_mean_gpu = cu(interpolated_pos);
        pos_cov_gpu = cu(interpolated_cov);
        sim_result_gpu = cu(sim_result);
        human_size_gpu = cu(cnt_param.human_size);
        epsilon_gpu = cu(cnt_param.epsilon);
        out = CuArray{Float32, 2}(undef, size(sim_result, 1), size(sim_result, 2));

        threads = threads;
        numblocks_x = ceil(Int, size(out, 1)/threads[1]);
        numblocks_y = ceil(Int, size(out, 2)/threads[2]);
        blocks = (numblocks_x, numblocks_y)
        CUDA.@sync begin
            @cuda threads=threads blocks=blocks kernel_CVaR!(out, pos_mean_gpu, pos_cov_gpu, sim_result_gpu, human_size_gpu, epsilon_gpu)
        end

        key_CVaR = collect(out);
        CVaR = max.(CVaR, key_CVaR);
    end

    max_CVaR = maximum(CVaR, dims=2);

    discount_factor = cumprod(0.9*ones(size(sim_result, 2)));
    sum_CVaR = sum(CVaR .* discount_factor, dims=2);

    return sum_CVaR, max_CVaR
end

function kernel_CVaR!(out::AbstractArray{Float32, 2},
                     predictive_mean::AbstractArray{Float32, 2},
                     predictive_cov::AbstractArray{Float32, 3},
                     ego_position::AbstractArray{Float32, 3},
                     human_size::Float32,
                     epsilon::Float32)
    # out(CVaR) : (n_controls, n_horizon)
    # predictive_mean : (n_controls, n_horizon, 2)
    # predictive_cov : (n_controls, n_horizon, 2, 2)
    # ego_position : (n_controls, n_horizon, 2)

    ii = (blockIdx().x - 1)*blockDim().x + threadIdx().x; # dimension for control candidates
    jj = (blockIdx().y - 1)*blockDim().y + threadIdx().y; # dimension for horizon
    if (ii <= size(out, 1)) && (jj <= size(out, 2))
        # get mean and cov
        mean = predictive_mean[ii, jj, :];
        cov = predictive_cov[ii, jj, :, :];
        # get ego position
        e_position = ego_position[ii, jj, :];
        # relative vector to the robot position from the human position
        rel_vec = e_position - mean;
        # compute distance between mean and ego agent
        dist = norm(rel_vec) - human_size;
        if dist > 0.0
            # Find the ellipsoid
            # (x - p_human)^T E (x - p_human) = 1 & E = Q D Q^t
            R = max(100.0, dist);
            D = diagm([1/dist^2, 1/R^2]);
            Q = [rel_vec[1]/dist rel_vec[2]/dist; -rel_vec[2]/dist rel_vec[1]/dist];
            E = Q*D*transpose(Q);
            # compute CVaR
            out[ii, jj] = -1 + 1/epsilon * tr(cov * E);
        else
            out[ii, jj] = 1.0;
        end
    end
    end
    return nothing
end

function compute_CVaR(sim_result::Vector{RobotState},
                            w_init::WorldState,
                            cnt_param::DRCControlParameter,
                            prediction_mean_dict::Dict{String, Array{Float64, 2}},
                            prediction_cov_dict::Dict{String, Array{Float64, 3}},
                            predict_idx::Vector{Int64},
                            pred_expansion_factor::Int64);

    CVaR = -1.0 .* ones(length(sim_result));

    current_ado_position_dict = w_init.ap_dict;

    for key in keys(prediction_mean_dict)
        # get mean and cov
        current_pos = current_ado_position_dict[key]';
        mean = prediction_mean_dict[key];
        pos = vcat(current_pos, mean);
        interpolated_pos = Array{Float64, 2}(undef, length(sim_result), 2);
        for i in 1:length(sim_result)-1
            interpolate = (rem(i, pred_expansion_factor)/pred_expansion_factor)*pos[div(i, pred_expansion_factor)+2,:] + 
                                (1-(rem(i, pred_expansion_factor)/pred_expansion_factor))*pos[div(i, pred_expansion_factor)+1,:];
            interpolated_pos[i, :] = interpolate;
        end
        interpolated_pos[end, :] = pos[end, :];

        cov = prediction_cov_dict[key];
        for (euler_idx, pred_idx) in enumerate(predict_idx)
            e_position = get_position(sim_result[euler_idx]);
            # relative vector to the robot position from the human position
            rel_vec = e_position - interpolated_pos[euler_idx, :];
            # compute distance between mean and ego agent
            dist = norm(rel_vec) - cnt_param.human_size;
            if dist > 0.0
                # Find the ellipsoid
                # (x - p_human)^T E (x - p_human) = 1 & E = Q D Q^t
                R = maximum([100.0, dist]);
                D = diagm([1/dist^2, 1/R^2]);
                Q = [rel_vec[1]/dist rel_vec[2]/dist; -rel_vec[2]/dist rel_vec[1]/dist];
                E = Q*D*transpose(Q);
                # compute CVaR
                CVaR[euler_idx] = max(CVaR[euler_idx], -1 + 1/cnt_param.epsilon * tr(cov[pred_idx, :, :] * E));
                # append!(CVaR, -1 + 1/cnt_param.epsilon * tr(cov[pred_idx, :, :] * E));
            else
                CVaR[euler_idx] = 1.0;
            end
        end
    end

    for idx in 1:length(sim_result)
        CVaR[idx] = CVaR[idx] * 0.9^(idx-1);
    end
    # if isempty(CVaR)
    #     return -100.0
    # else
    #     return maximum(CVaR)
    # end
    return sum(CVaR), maximum(CVaR)
end

# for Trajectron robot-future-conditoinal models
function get_robot_present_and_future(e_init::RobotState,
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
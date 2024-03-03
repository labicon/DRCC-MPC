#///////////////////////////////////////
#// File Name: utils.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/02/25
#// Description: Utility functions for Risk Sensitive SAC
#///////////////////////////////////////

import StatsBase: fit, Histogram
import PlotUtils: plot_color

using Distributions
using FileIO
using JLD2
using Plots
using Printf
using ProgressMeter
using Random
using RobotOS

# convert u_nominal to u_schedule (ordereddict)
function convert_to_schedule(t_init::Time, u_nominal::Vector{Float64},
                             sim_param::SimulationParameter)
    u_schedule = OrderedDict{Time, Vector{Float64}}();
    plan_horizon = sim_param.prediction_steps*sim_param.dto;
    t = t_init;
    while t < Time(to_sec(t_init) + plan_horizon)
        u_schedule[t] = u_nominal;
        t += Duration(sim_param.dtc);
    end
    return u_schedule
end

# compute nominal trajectory
@inline function get_nominal_trajectory(e_init::Union{RobotState, UnicycleState},
                                e_pos_goal_vec::Vector{Float64},
                                target_speed::Float64,
                                sim_horizon::Float64,
                                prediction_horizon::Float64)
    target_horizon = norm(e_pos_goal_vec - get_position(e_init))/target_speed;
    points = [WayPoint2D(get_position(e_init), e_init.t),
              WayPoint2D(e_pos_goal_vec, e_init.t + Duration(target_horizon))];
    if target_horizon <= sim_horizon + prediction_horizon
        push!(points, WayPoint2D(e_pos_goal_vec,
                                 e_init.t + Duration(sim_horizon) +
                                            Duration(prediction_horizon)));
    end
    return Trajectory2D(points)
end

function init_condition_setup(;# Ego Initial Conditions
                              ego_pos_init_vec::Vector{Float64},
                              ego_vel_init_vec::Union{Nothing, Vector{Float64}}=nothing,
                              ego_pos_goal_vec::Vector{Float64},
                              t_init::Time=Time(0.0),
                              # Ado Initial Positions
                              ado_positions::Dict{String, Vector{Float64}},
                              # Other Parameters
                              target_speed::Union{Nothing, Float64}=nothing,
                              sim_horizon::Float64,
                              sim_param::SimulationParameter,
                              target_trajectory_required::Bool=true)
    # Initial condition setup
    if isnothing(ego_vel_init_vec)
        # @assert !isnothing(target_speed)
        ego_vel_init_vec = (ego_pos_goal_vec - ego_pos_init_vec[1:2])./
                           norm(ego_pos_goal_vec - ego_pos_init_vec[1:2]);
    end
    ego_state_init_vec = vcat(ego_pos_init_vec, ego_vel_init_vec);
    if length(ego_state_init_vec) == 4
        e_init = RobotState(ego_state_init_vec, t_init);
    elseif length(ego_state_init_vec) == 5
        e_init = UnicycleState(ego_state_init_vec, t_init);
    end
    w_init = WorldState(e_init, ado_positions, t_init);

    # Measurement schedule setup
    measurement_schedule =
        get_measurement_schedule(w_init, sim_horizon, sim_param);

    if target_trajectory_required
        @assert !isnothing(target_speed)
        # Target Trajectory setup
        prediction_horizon = sim_param.dto*sim_param.prediction_steps;
        target_trajectory = get_nominal_trajectory(e_init,
                                                   ego_pos_goal_vec,
                                                   target_speed,
                                                   sim_horizon,
                                                   prediction_horizon);
        return w_init, measurement_schedule, target_trajectory
    else
        return w_init, measurement_schedule
    end
end

# DRC and Trajectron controller setup
function controller_setup(# Scene Loader parameters
                            scene_param::TrajectronSceneParameter,
                            # Predictor Parameters
                            predictor_param::TrajectronPredictorParameter;
                            prediction_device::String,
                            # Cost Parameters
                            cost_param::DRCCostParameter,
                            # Control Parameters
                            cnt_param::DRCControlParameter,
                            # Simulation Parameters
                            dtc::Float64,
                            # Ego Initial Conditions
                            ego_pos_init_vec::Union{Nothing, Vector{Float64}}=nothing,
                            ego_vel_init_vec::Union{Nothing, Vector{Float64}}=nothing,
                            ego_pos_goal_vec::Union{Nothing, Vector{Float64}}=nothing,
                            t_init::Time=Time(0.0),
                            # Other Parameters
                            target_speed::Union{Nothing, Float64}=nothing,
                            sim_horizon::Float64,
                            ado_id_to_replace::Union{Nothing, String}=nothing,
                            verbose=true)
    if verbose
        println("Scene Mode: data")
        println("Prediction Mode: trajectron")
        println("Deterministic Prediction: $(predictor_param.deterministic)")
    end
    if prediction_device == "cpu"
        device = py"torch".device("cpu");
    elseif prediction_device == "cuda"
        device = py"torch".device("cuda");
    end
    scene_loader = TrajectronSceneLoader(scene_param, verbose=verbose,
                    ado_id_removed=ado_id_to_replace);
    if !isnothing(ado_id_to_replace)
        @assert isnothing(ego_pos_init_vec) && isnothing(ego_vel_init_vec) &&
                    isnothing(ego_pos_goal_vec) && isnothing(target_speed)
        target_trajectory = get_trajectory_for_ado(scene_loader,
                                    t_init,
                                    ado_id_to_replace,
                                    sim_horizon);
        t_init_new = minimum(target_trajectory)[1];
        t_end = maximum(target_trajectory)[1];
        timesteps_forward = Int64(round(to_sec(t_init_new - t_init)/scene_loader.dto, digits=5));
        println("Found $(ado_id_to_replace). start_time_idx updated to: $(scene_param.start_time_idx + timesteps_forward) Re-loading Scene...")
        sim_horizon -= to_sec(t_init_new - t_init);
        ego_pos_init_vec = minimum(target_trajectory)[2];
        ego_vel_init_vec = (collect(values(target_trajectory))[2] - ego_pos_init_vec)./scene_loader.dto;
        ego_pos_goal_vec = maximum(target_trajectory)[2];
        scene_param_new = TrajectronSceneParameter(scene_param.conf_file_name,
                                scene_param.test_data_name,
                                scene_param.test_scene_id,
                                scene_param.start_time_idx + timesteps_forward,
                                scene_param.incl_robot_node);
        scene_loader = TrajectronSceneLoader(scene_param_new, verbose=verbose,
                            ado_id_removed=ado_id_to_replace);
        target_trajectory = get_trajectory_for_ado(scene_loader,
                                    t_init_new,
                                    ado_id_to_replace,
                                    sim_horizon);
        prediction_horizon = scene_loader.dto*predictor_param.prediction_steps;
        if to_sec(t_end - t_init_new) < sim_horizon + prediction_horizon
            target_trajectory[t_init_new + Duration(sim_horizon + prediction_horizon)] = ego_pos_goal_vec;
        end
    end
    predictor = TrajectronPredictor(predictor_param,
                scene_loader.model_dir,
                scene_loader.param.conf_file_name,
                device, verbose=verbose);
    initialize_scene_graph!(predictor, scene_loader);
    sim_param = SimulationParameter(scene_loader, predictor, dtc, cost_param);

    ado_inputs = fetch_ado_positions!(scene_loader, return_full_state=true);
    ado_positions = reduce_to_positions(ado_inputs);

    if !isnothing(ado_id_to_replace)
        key_to_remove = nothing
        for key in keys(ado_positions)
            if pybuiltin("str")(key) == ado_id_to_replace
                key_to_remove = key
            end
        end
        delete!(ado_positions, key_to_remove)
        delete!(ado_inputs, key_to_remove)

        println("Note initial time is set to $(to_sec(t_init_new)) [s].")
        w_init, measurement_schedule =
            init_condition_setup(ego_pos_init_vec=ego_pos_init_vec,
                    ego_vel_init_vec=ego_vel_init_vec,
                    ego_pos_goal_vec=ego_pos_goal_vec,
                    t_init=t_init_new,
                    ado_positions=convert_nodes_to_str(ado_positions),
                    sim_horizon=sim_horizon,
                    sim_param=sim_param,
                    target_trajectory_required=false);
    else
        @assert !isnothing(ego_pos_init_vec) && !isnothing(ego_pos_goal_vec)
        w_init, measurement_schedule, target_trajectory =
            init_condition_setup(ego_pos_init_vec=ego_pos_init_vec,
                    ego_vel_init_vec=ego_vel_init_vec,
                    ego_pos_goal_vec=ego_pos_goal_vec,
                    t_init=t_init,
                    ado_positions=convert_nodes_to_str(ado_positions),
                    target_speed=target_speed,
                    sim_horizon=sim_horizon,
                    sim_param=sim_param,
                    target_trajectory_required=true);
    end
    # Controller setup
    controller = DRCController(sim_param, cnt_param, predictor, cost_param);
    if predictor.param.use_robot_future
        schedule_prediction!(controller, ado_inputs, w_init.e_state);
    else
        schedule_prediction!(controller, ado_inputs);
    end
    wait(controller.prediction_task);

    return scene_loader, controller, w_init, measurement_schedule, target_trajectory, target_speed
end

function controller_setup(# Scene Loader parameters
                            scene_param::SyntheticSceneParameter,
                            # Predictor Parameters
                            predictor_param::GaussianPredictorParameter;
                            prediction_device::String,
                            # Cost Parameters
                            cost_param::DRCCostParameter,
                            # Control Parameters
                            cnt_param::DRCControlParameter,
                            # Simulation Parameters
                            dtc::Float64,
                            dto::Float64,
                            ado_pos_init_dict::Dict{String, Vector{Float64}},
                            ado_vel_dict::Dict{String, <:MvNormal},
                            # Ego Initial Conditions
                            ego_pos_init_vec::Vector{Float64},
                            ego_vel_init_vec::Union{Nothing, Vector{Float64}}=nothing,
                            ego_pos_goal_vec::Vector{Float64},
                            t_init::Time=Time(0.0),
                            # Other Parameters
                            target_speed::Float64,
                            sim_horizon::Float64,
                            verbose=true)
    if verbose
        println("Scene Mode: synthetic")
        println("Prediction Mode: gaussian")
        println("Deterministic Prediction: $(predictor_param.deterministic)")
    end
    if prediction_device == "cuda"
        @warn "Prediction on CUDA devices is not currently supported. Using CPU."
    end
    device = py"torch".device("cpu");

    scene_loader = SyntheticSceneLoader(scene_param,
                    deepcopy(ado_pos_init_dict));
    predictor = GaussianPredictor(predictor_param, dto, ado_vel_dict);
    sim_param = SimulationParameter(predictor, dtc, cost_param);

    w_init, measurement_schedule, target_trajectory =
                        init_condition_setup(ego_pos_init_vec=ego_pos_init_vec,
                            ego_vel_init_vec=ego_vel_init_vec,
                            ego_pos_goal_vec=ego_pos_goal_vec,
                            t_init=t_init,
                            ado_positions=ado_pos_init_dict,
                            target_speed=target_speed,
                            sim_horizon=sim_horizon,
                            sim_param=sim_param);
    # Controller setup
    controller = DRCController(sim_param, cnt_param, predictor, cost_param);
    schedule_prediction!(controller, ado_pos_init_dict);
    wait(controller.prediction_task);

    return scene_loader, controller, w_init, measurement_schedule, target_trajectory, target_speed
end

function controller_setup(# Scene Loader parameters
        scene_param::StopSyntheticSceneParameter,
        # Predictor Parameters
        predictor_param::StopGaussianPredictorParameter;
        prediction_device::String,
        # Cost Parameters
        cost_param::DRCCostParameter,
        # Control Parameters
        cnt_param::DRCControlParameter,
        # Simulation Parameters
        dtc::Float64,
        dto::Float64,
        ado_pos_init_dict::Dict{String, Vector{Float64}},
        ado_vel_dict::Dict{String, Vector{DiagNormal}},
        # Ego Initial Conditions
        ego_pos_init_vec::Vector{Float64},
        ego_vel_init_vec::Union{Nothing, Vector{Float64}}=nothing,
        ego_pos_goal_vec::Vector{Float64},
        t_init::Time=Time(0.0),
        # Other Parameters
        target_speed::Float64,
        sim_horizon::Float64,
        verbose=true)
    if verbose
        println("Scene Mode: synthetic")
        println("Prediction Mode: gaussian")
        println("Deterministic Prediction: $(predictor_param.deterministic)")
    end
    if prediction_device == "cuda"
        @warn "Prediction on CUDA devices is not currently supported. Using CPU."
    end
    device = py"torch".device("cpu");

    scene_loader = StopSyntheticSceneLoader(scene_param, deepcopy(ado_pos_init_dict), false);
    predictor = StopGaussianPredictor(predictor_param, dto, ado_vel_dict);
    sim_param = SimulationParameter(predictor, dtc, cost_param);

    w_init, measurement_schedule, target_trajectory =
        init_condition_setup(ego_pos_init_vec=ego_pos_init_vec,
            ego_vel_init_vec=ego_vel_init_vec,
            ego_pos_goal_vec=ego_pos_goal_vec,
            t_init=t_init,
            ado_positions=ado_pos_init_dict,
            target_speed=target_speed,
            sim_horizon=sim_horizon,
            sim_param=sim_param);
    # Controller setup
    controller = DRCController(sim_param, cnt_param, predictor, cost_param);
    schedule_prediction!(controller, ado_pos_init_dict);
    wait(controller.prediction_task);

    return scene_loader, controller, w_init, measurement_schedule, target_trajectory, target_speed
end

function controller_setup(# Scene Loader parameters
                            scene_param::ShiftedSyntheticSceneParameter,
                            # Predictor Parameters
                            predictor_param::GaussianPredictorParameter;
                            prediction_device::String,
                            # Cost Parameters
                            cost_param::DRCCostParameter,
                            # Control Parameters
                            cnt_param::DRCControlParameter,
                            # Simulation Parameters
                            dtc::Float64,
                            dto::Float64,
                            ado_pos_init_dict::Dict{String, Vector{Float64}},
                            ado_vel_dict::Dict{String, <:MvNormal},
                            ado_true_vel_dict::Dict{String, Vector{DiagNormal}},
                            # Ego Initial Conditions
                            ego_pos_init_vec::Vector{Float64},
                            ego_vel_init_vec::Union{Nothing, Vector{Float64}}=nothing,
                            ego_pos_goal_vec::Vector{Float64},
                            t_init::Time=Time(0.0),
                            # Other Parameters
                            target_speed::Float64,
                            sim_horizon::Float64,
                            verbose=true)
    if verbose
        println("Scene Mode: synthetic")
        println("Prediction Mode: gaussian")
        println("Deterministic Prediction: $(predictor_param.deterministic)")
    end
    if prediction_device == "cuda"
        @warn "Prediction on CUDA devices is not currently supported. Using CPU."
    end
    device = py"torch".device("cpu");

    scene_loader = ShiftedSyntheticSceneLoader(scene_param,
                                                deepcopy(ado_pos_init_dict),
                                                ado_true_vel_dict);
    predictor = GaussianPredictor(predictor_param, dto, ado_vel_dict);
    sim_param = SimulationParameter(predictor, dtc, cost_param);

    w_init, measurement_schedule, target_trajectory =
    init_condition_setup(ego_pos_init_vec=ego_pos_init_vec,
                            ego_vel_init_vec=ego_vel_init_vec,
                            ego_pos_goal_vec=ego_pos_goal_vec,
                            t_init=t_init,
                            ado_positions=ado_pos_init_dict,
                            target_speed=target_speed,
                            sim_horizon=sim_horizon,
                            sim_param=sim_param);
    # Controller setup
    controller = DRCController(sim_param, cnt_param, predictor, cost_param);
    schedule_prediction!(controller, ado_pos_init_dict);
    wait(controller.prediction_task);

    return scene_loader, controller, w_init, measurement_schedule, target_trajectory, target_speed
end

# display log
function display_log(log::Vector{Tuple{Time, String}})
    for l in log
        time = @sprintf "Time %.2f" round(to_sec(l[1]), digits=5)
        println("$(time) [sec]: $(l[2])");
    end
end


# plot helper functions
#=
function visualize!(color_dict::Dict, ap_dict::Dict{String, Vector{Float64}},
                    outputs_dict::Dict{String, Array{Float64, 3}};
                    xlim=(0., 20.), ylim=(0., 20.), markersize=8.0)
    plt = plot(xlim=xlim, ylim=ylim, aspect_ratio=1.0, xlabel="x[m]",
               ylabel="y[m]");
    for key in keys(ap_dict)
        if !haskey(color_dict, key)
            color_dict[key] = palette(:default)[length(color_dict) + 1];
        end
        scatter!((ap_dict[key][1], ap_dict[key][2]),
                 color=color_dict[key], label="", #label=key,
                 markersize=markersize/1.5, markershape=:rect);
        if haskey(outputs_dict, key)
            for jj = 1:size(outputs_dict[key], 1)
                plot!(outputs_dict[key][jj, :, 1], outputs_dict[key][jj, :, 2],
                      color=color_dict[key], label="", alpha=0.3)
            end
        end
        #jj = 5
        #plot!(outputs_array[ii, jj, :, 1], outputs_array[ii, jj, :, 2],
        #      color=color_dict[node_names_and_pos[ii][1]], label="", alpha=0.3)
    end
    return plt
end
=#

function visualize!(color_dict::Dict, w::WorldState,
                    target_trajectory::Trajectory2D,
                    outputs_dict::Dict{String, Array{Float64, 3}},
                    num_samples::Int64,
                    #nominal_control_idx::Int64,
                    #nominal_trajectory::Union{Nothing, Vector{Vector{Float64}}}=nothing;
                    figsize, legend,
                    legendfontsize,
                    xlim, ylim, markersize,
                    show_velocity,
                    show_prediction,
                    dummy_pos)
    plt = plot(legendfontsize=legendfontsize,
               legend=legend, size=figsize);
    plot!(xlim=xlim, ylim=ylim, aspect_ratio=1.0, xlabel="x[m]", ylabel="y[m]");
    robot_key = "Ego Robot"
    # palette = get_color_palette(:auto, plot_color(:white), 30)
    palette = get_color_palette(:tab20, 30)
    if !haskey(color_dict, robot_key)
        color_dict[robot_key] = palette[1];
    end
    t = @sprintf "Time: %.2f [s]" round(to_sec(w.t), digits=2)
    scatter!((w.e_state.x[1], w.e_state.x[2]), color=color_dict[robot_key],
             label=robot_key,
             markersize=markersize,
             title=t)
    if show_velocity
        pos = get_position(w.e_state);
        vel = get_velocity(w.e_state);
        plot!([pos[1], pos[1] + vel[1]],
              [pos[2], pos[2] + vel[2]],
              color=color_dict[robot_key], label="Velocity", arrow=:arrow,
              alpha=0.5)
    end
    goal_pos = collect(values(target_trajectory))[end];
    scatter!((goal_pos[1], goal_pos[2]), color=color_dict[robot_key],
             label="Ego Goal",
             markershape=:star5,
             markersize=markersize)
    for key in keys(w.ap_dict)
        if !haskey(color_dict, key)
            color_dict[key] = palette[length(color_dict) + 1];
        end
        scatter!((w.ap_dict[key][1], w.ap_dict[key][2]),
                 color=color_dict[key], label="", #label=key,
                 markersize=markersize/1.5, markershape=:rect);
        if show_prediction && haskey(outputs_dict, key)
            # note that prediction is dependent on the nominal control choice.
            prediction_array = outputs_dict[key]
            if isnothing(dummy_pos)
                for jj = 1:size(prediction_array, 1)
                    plot!(prediction_array[jj, :, 1], prediction_array[jj, :, 2],
                          color=color_dict[key], label="", alpha=0.3)
                end
            else
                for jj = 1:size(prediction_array, 1)
                    prediction_array_filtered = [hcat(filter(x -> x != dummy_pos, prediction_array[jj, :, 1])...);
                                                 hcat(filter(y -> y != dummy_pos, prediction_array[jj, :, 2])...)];
                    if size(prediction_array_filtered, 1) == 2 &&
                       length(prediction_array_filtered[1, :]) == length(prediction_array_filtered[2, :])
                       plot!(prediction_array_filtered[1, :], prediction_array_filtered[2, :],
                             color=color_dict[key], label="", alpha=0.3)
                    end
                end
            end
        end
        #jj = 5
        #plot!(outputs_array[ii, jj, :, 1], outputs_array[ii, jj, :, 2],
        #      color=color_dict[node_names_and_pos[ii][1]], label="", alpha=0.3)
    end
    return plt
end

function make_gif(result::DRCEvaluationResult;
                  dtplot::Float64,
                  fps::Int64,
                  figsize::Tuple{Int64, Int64},
                  legendfontsize::Int64,
                  legend::Symbol,
                  xlim::Tuple{Float64,Float64},
                  ylim::Tuple{Float64,Float64},
                  markersize::Float64,
                  filename::String,
                  show_prediction=true,
                  show_nominal_trajectory=false,
                  show_past_ego_trajectory=true,
                  dummy_pos=nothing)
    anim = Animation();
    color_dict = Dict();
    prediction_steps = result.sim_param.prediction_steps;
    dtc = result.sim_param.dtc;
    dto = result.sim_param.dto;
    plan_horizon = dto*prediction_steps;
    @assert dtplot >= dtc;
    dtplot = dtplot;
    if typeof(result) == DRCEvaluationResult
        prediction_dict_last = nothing;
    end
    @showprogress for ii = 1:length(result.w_history)
        w = result.w_history[ii]
        if round(to_sec(w.t)/dtplot) ≈ to_sec(w.t)/dtplot
            if isnothing(result.prediction_dict_history[ii])
                prediction_dict = prediction_dict_last;
            else
                prediction_dict = result.prediction_dict_history[ii];
                prediction_dict_last = prediction_dict;
            end
            #if ii == length(result.w_history)
            #    u_nominal_idx = result.u_nominal_idx_history[ii - 1]
            #else
            #    u_nominal_idx = result.u_nominal_idx_history[ii]
            #end
            if show_nominal_trajectory
                nominal_trajectory = result.nominal_trajectory_history[ii]
            else
                nominal_trajectory = nothing
            end
            plt = visualize!(color_dict, w,
                             result.target_trajectory_history[ii],
                             prediction_dict,
                             result.sim_param.num_samples,
                             # u_nominal_idx,
                             # nominal_trajectory,
                             figsize, legend,
                             legendfontsize,
                             xlim, ylim, markersize,
                             true,
                             show_prediction,
                             dummy_pos)
            if show_past_ego_trajectory
                ego_traj_x = [get_position(w.e_state)[1] for w in result.w_history[1:ii]]
                ego_traj_y = [get_position(w.e_state)[2] for w in result.w_history[1:ii]]
                ego_color = color_dict["Ego Robot"];
                plot!(ego_traj_x, ego_traj_y, color=ego_color, label="Ego Trajectory", linewidth=1.5)
            end
            frame(anim, plt)
        end
    end
    gif(anim, filename, fps=fps);
end

function fetch_stats_filtered(func::Function...;
                              file_dir::String,
                              filtering_str::String)
    files = readdir(file_dir)
    valid_files = filter(file -> (occursin(".jld2", file) && occursin(filtering_str, file)), files)
    println("Loading $(length(valid_files)) jld2 files with '$(filtering_str)'.")

    return_array = [Real[] for ii = 1:length(func)]

    @showprogress for file in valid_files
        file_path = joinpath(file_dir, file);
        file_content = load(file_path);
        result = file_content["result"]

        for ii = 1:length(func)
            push!(return_array[ii], (func[ii])(result))
        end

        close(file_path)
        file_content = nothing;
    end
    return return_array
end

function plot_histogram(data_array...; min_val::Real, max_val::Real, num_bins::Int64,
                        color::Vector{Symbol}, alpha::Real,
                        xlabel="", ylabel="", label = ["" for ii = 1:length(data_array)],
                        title="")
    plt = plot(xlabel=xlabel, ylabel=ylabel, title=title);

    edges = range(min_val, stop=max_val, length=num_bins+1);
    offset = 0.0
    for ii = 1:length(data_array)
        hist = fit(Histogram, data_array[ii], edges .+ offset)
        delta = (edges[2] - edges[1])/length(data_array);
        bar!(hist.edges[1][1:end-1], hist.weights, bar_width=delta, alpha=alpha,
             color=color[ii], label=label[ii])
        offset += delta
    end
    #=
    for ii = 1:length(data_array)
        histogram!(data_array[ii], xlim=(min_val, max_val), bins=min_val:(max_val-min_val)/num_bins:max_val,
                    title=title, xlabel=xlabel, label=label[ii], alpha=alpha);
    end
    =#
    return plt
end

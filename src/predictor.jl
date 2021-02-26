#///////////////////////////////////////
#// File Name: predictor.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/01/28
#// Description: Ado prediction models for Risk Sensitive Stochastic SAC
#///////////////////////////////////////

using LinearAlgebra
using PyCall
using Distributions

abstract type Predictor end;

# Trajectron Predictor
struct TrajectronPredictorParameter <: Parameter
    prediction_steps::Int64    # Prediction steps for ado robots (Trajectron)
    num_samples::Int64         # Number of samples for ado simulation (Trajectron)
    use_robot_future::Bool     # Whether Trajectron is conditioned on the robot future.
    deterministic::Bool        # Use mode-mode sampling for deterministic prediction.
    rng_seed_py::Int64         # rng seed (Trajectron)
end

mutable struct TrajectronPredictor <: Predictor
    param::TrajectronPredictorParameter
    trajectron::PyObject       # Trajectron model
    edge_addition_filter::PyObject
    edge_removal_filter::PyObject
end

function TrajectronPredictor(param::TrajectronPredictorParameter,
                             model_dir::String,
                             conf_file::String,
                             device::PyObject;
                             verbose=true)
    if !occursin("rob", model_dir)
        incl_robot_node = false;
    else
        incl_robot_node = true;
    end
    if param.deterministic
        @assert param.num_samples == 1 "num_samples has to be 1 for deterministic prediction!"
    end

    # Trajectron arguments
    py"""
    args = dict({
        # Model Parameters
        "offline_scene_graph": "no",                    # whether to precompute the scene graphs offline, options are "no" and "yes"
        "dynamic_edges": "yes",                         # whether to use dynamic edges or not, options are "no" and "yes"
        "edge_state_combine_method": "sum",             # the method to use for combining edges of the same type
        "edge_influence_combine_method": "attention",   # the method to use for combining edge influences
        "edge_addition_filter": [0.25, 0.5, 0.75, 1.0], # what scaling to use for edges as they're created
        "edge_removal_filter": [1.0, 0.0],              # what scaling to use for edges as they're removed
        "incl_robot_node": $(incl_robot_node),          # whether to include a robot node in the graph or simply model all agents
        "map_encoding": False,                          # whether to use map encoding or not
        "no_edge_encoding": False,                      # whether to use neighbors edge encoding
        "eval_device": $(device),                       # what device to use during evaluation
        "seed": $(param.rng_seed_py),                   # manual seed to use
    })
    """
    trajectron, hyperparams, log =
        pycall(py"initialize_model", NTuple{3, PyObject},
               pycall(py"EasyDict", PyObject, py"args"), model_dir, conf_file);

    if verbose
        for string in log
            println(string)
        end
    end

    return TrajectronPredictor(param, trajectron,
                               pycall(hyperparams.get, PyObject, "edge_addition_filter"),
                               pycall(hyperparams.get, PyObject, "edge_removal_filter"))
end

function initialize_scene_graph!(predictor::TrajectronPredictor,
                                 scene_loader::TrajectronSceneLoader)

    if predictor.param.use_robot_future
        @assert scene_loader.param.incl_robot_node "Using robot future without spawning robot in the scene!"
    end
    if scene_loader.param.incl_robot_node
        @assert predictor.param.use_robot_future "Spawning robot in the scene without using robot future!"
    end
    scene_loader.online_scene.calculate_scene_graph(attention_radius=scene_loader.eval_env.attention_radius,
                                                    edge_addition_filter=predictor.edge_addition_filter,
                                                    edge_removal_filter=predictor.edge_removal_filter);
    online_env = py"Environment"(node_type_list=scene_loader.eval_env.node_type_list,
                                 standardization=scene_loader.eval_env.standardization,
                                 scenes=[scene_loader.online_scene],
                                 attention_radius=scene_loader.eval_env.attention_radius,
                                 robot_type=scene_loader.eval_env.robot_type)
    predictor.trajectron.set_environment(online_env, scene_loader.curr_time_idx - 1)
end

function sample_future_ado_positions!(predictor::TrajectronPredictor,
                                      ado_state_dict::Dict,
                                      robot_present_and_future::Union{Nothing, Array{Float64, 3}}=nothing)
    z_mode, gmm_mode = false, false
    if predictor.param.deterministic
        z_mode, gmm_mode = true, true # mode-mode sampling
    end
    if !isnothing(robot_present_and_future)
        @assert predictor.param.use_robot_future "Do not provide robot_present_and_future when not using it!"
    end
    if predictor.param.use_robot_future
        @assert !isnothing(robot_present_and_future) "robot_present_and_future must not be nothing!"
    end
    if !isnothing(robot_present_and_future)
        state_dict = Dict{PyObject, Vector{Float64}}();
        robot_state = robot_present_and_future[1, 1, :];
        robot_node = predictor.trajectron.env.scenes[1].robot;
        state_dict[robot_node] = robot_state;
        for key in keys(ado_state_dict)
            state_dict[key] = ado_state_dict[key]
        end
    else
        state_dict = ado_state_dict;
    end
    maps = nothing; # TODO: Implement maps if necessary.
    preds_py = predictor.trajectron.incremental_forward(state_dict, maps=maps,
                                                        prediction_horizon=predictor.param.prediction_steps,
                                                        num_samples=predictor.param.num_samples,
                                                        z_mode=z_mode,
                                                        gmm_mode=gmm_mode,
                                                        robot_present_and_future=robot_present_and_future)

    preds_py = preds_py[2]; # using samples only
    outputs_dict = Dict{String, Array{Float64, 3}}();
    for key in keys(preds_py)
        if isnothing(robot_present_and_future) && predictor.param.use_robot_future
            outputs_dict[pybuiltin("str")(key)] = preds_py[key][1, :, :, :]
        else
            outputs_dict[pybuiltin("str")(key)] =
                reshape(permutedims(preds_py[key], [2, 1, 3, 4]), :,
                        size(preds_py[key], 3), size(preds_py[key], 4))
        end
    end
    return outputs_dict
end

# Oracle Predictor (for Data Scenes Only)
struct OraclePredictorParameter <: Parameter
    prediction_steps::Int64
    dummy_pos::Float64 # dummy variable to fill in missing prediction values
end

function OraclePredictorParameter(prediction_steps::Int64)
    dummy_pos = 500.;
    OraclePredictorParameter(prediction_steps, dummy_pos)
end

mutable struct OraclePredictor <: Predictor
    param::OraclePredictorParameter
    eval_scene::PyObject
    state_def::PyObject
    curr_time_idx::Int64
    max_time_idx::Int64
end

function OraclePredictor(param::OraclePredictorParameter,
                         scene_loader::TrajectronSceneLoader)
    @assert scene_loader.curr_time_idx + param.prediction_steps <= scene_loader.max_time_idx
    return OraclePredictor(param, scene_loader.eval_scene, scene_loader.state_def,
                           scene_loader.curr_time_idx, scene_loader.max_time_idx);
end

function sample_future_ado_positions!(predictor::OraclePredictor,
                                      ado_pos_dict::Dict)
    # existing ado agents
    full_ado_list =
        predictor.eval_scene.get_nodes_clipped_at_time(collect(predictor.curr_time_idx:
                                                               predictor.curr_time_idx + predictor.param.prediction_steps),
                                                       predictor.state_def);
    # initialize outputs_dict
    outputs_dict = Dict{String, Array{Float64, 3}}();
    for key in full_ado_list
        if !in(key, keys(ado_pos_dict))
            continue;
        end
        outputs_dict[pybuiltin("str")(key)] =
            Array{Float64, 3}(undef, 1, predictor.param.prediction_steps, 2);
    end

    # sweep through pos_dict at different timesteps and fill in outputs_dict
    for ii = 1:predictor.param.prediction_steps
        input_dict = predictor.eval_scene.get_clipped_input_dict(predictor.curr_time_idx + ii,
                                                                 predictor.state_def)
        input_dict = convert_nodes_to_str(input_dict);
        ado_intersect = intersect(keys(input_dict), keys(outputs_dict))
        for key in ado_intersect
            outputs_dict[key][1, ii, 1] = input_dict[key][1];
            outputs_dict[key][1, ii, 2] = input_dict[key][2];
        end
        ado_missing = setdiff(keys(outputs_dict), ado_intersect)
        for ado in ado_missing
            outputs_dict[ado][1, ii, 1] = predictor.param.dummy_pos
            outputs_dict[ado][1, ii, 2] = predictor.param.dummy_pos
        end
    end
    predictor.curr_time_idx += 1;
    return outputs_dict
end


# Gaussian Predictor
struct GaussianPredictorParameter <: Parameter
    #dto::Float64
    prediction_steps::Int64
    num_samples::Int64
    deterministic::Bool   # Use mean estimate for deterministic prediction
    rng::AbstractRNG
end

mutable struct GaussianPredictor <: Predictor
    param::GaussianPredictorParameter
    dto::Float64
    ado_vel_dict::Dict{String, <:MvNormal}

    function GaussianPredictor(param::GaussianPredictorParameter,
                               dto::Float64,
                               ado_vel_dict::Dict{String, <:MvNormal})
        if param.deterministic
            @assert param.num_samples == 1 "num_samples has to be 1 for deterministic prediction!"
        end
        return new(param, dto, ado_vel_dict)
    end
end

function sample_future_ado_positions!(predictor::GaussianPredictor,
                                      ado_pos_dict::Dict{String, Vector{Float64}})
    @assert keys(predictor.ado_vel_dict) == keys(ado_pos_dict) ||
            collect(keys(predictor.ado_vel_dict)) == ["Any"]

    outputs_dict = Dict{String, Array{Float64, 3}}();
    for key in keys(ado_pos_dict)
        outputs_dict[key] =
            Array{Float64, 3}(undef, predictor.param.num_samples,
                              predictor.param.prediction_steps, 2);
        if collect(keys(predictor.ado_vel_dict)) == ["Any"]
            d = predictor.ado_vel_dict["Any"];
        else
            d = predictor.ado_vel_dict[key];
        end
        if predictor.param.deterministic
            for ii = 1:predictor.param.num_samples
                outputs_dict[key][ii, :, :] =
                    Array(transpose(hcat(
                        [d.μ for ii = 1:predictor.param.prediction_steps]...
                    )));
            end
        else
            for ii = 1:predictor.param.num_samples
                outputs_dict[key][ii, :, :] =
                    Array(transpose(hcat(
                        [rand(predictor.param.rng, d) for ii = 1:predictor.param.prediction_steps]...
                    )));
            end
        end
        outputs_dict[key] = cumsum(outputs_dict[key], dims=2).*
                            predictor.dto;
        outputs_dict[key][:, :, 1] .+= ado_pos_dict[key][1];
        outputs_dict[key][:, :, 2] .+= ado_pos_dict[key][2];
    end
    return outputs_dict
end

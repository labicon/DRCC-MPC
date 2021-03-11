#///////////////////////////////////////
#// File Name: scene_loader.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/01/15
#// Description: Scene loaders to load ado positions
#///////////////////////////////////////

using PyCall

abstract type SceneLoader end

# Trajectron Scene Loader
struct TrajectronSceneParameter <: Parameter
    conf_file_name::String     # json config file name.
    test_data_name::String     # Specify "eth_test.pkl", "hotel_test.pkl", or "univ_test.pkl"
    test_scene_id::Int64       # Test Scene ID (Trajectron) (starting from 0)
    start_time_idx::Int64      # Simulation start time index (Trajectron) (starting from 0, but you should start from 2)
    incl_robot_node::Bool      # Whether or not the robot node is included in the scene.
end

mutable struct TrajectronSceneLoader <: SceneLoader
    param::TrajectronSceneParameter
    model_dir::String              # model directory to be loaded in Trajectron Predictor
    eval_env::PyObject             # evaluation Environment object (Trajectron)
    eval_scene::PyObject           # evaluation Scene object (Trajectron) (i.e. entire scene sequence)
    online_scene::PyObject         # online Scene object (Trajectron) (i.e. current scene)
    state_def::PyObject                # state definitions of agents in the scene
    dto::Float64
    curr_time_idx::Int64           # Simulation current time index (Trajectron)
    max_time_idx::Int64            # Simulation max time index (Trajectron)
end

function TrajectronSceneLoader(param::TrajectronSceneParameter;
                               verbose=true,
                               ado_id_removed=nothing)
   # note removed ado agent still exists in eval_scene and
   # will be returned by fetch_ado_positions!. You'll have
   # to manually delete the ado agent from the outputs_dict.

    @assert in(param.test_data_name, ["eth", "hotel", "univ"] .* "_test.pkl");
    file_dir = @__DIR__;
    model_dir = normpath(joinpath(file_dir, "..",
                                  "Trajectron-plus-plus", "experiments", "pedestrians",
                                  "models"));
    if param.test_data_name == "eth_test.pkl"
        if param.incl_robot_node
            model_dir = joinpath(model_dir, "models_21_Apr_2020_20_15_41_eth_ar3_rob");
        else
            model_dir = joinpath(model_dir, "eth_attention_radius_3");
        end
    elseif param.test_data_name == "hotel_test.pkl"
        if param.incl_robot_node
            model_dir = joinpath(model_dir, "models_21_Apr_2020_20_15_59_hotel_ar3_rob")
        else
            model_dir = joinpath(model_dir, "hotel_attention_radius_3");
        end
    elseif param.test_data_name == "univ_test.pkl"
        if param.incl_robot_node
            model_dir = joinpath(model_dir, "models_21_Apr_2020_20_17_14_univ_ar3_rob");
        else
            model_dir = joinpath(model_dir, "univ_attention_radius_3");
        end
    end
    hyperparams = pycall(py"load_hyperparams", PyObject,
                         model_dir, param.conf_file_name);
    maximum_history_length = pycall(hyperparams.get, PyObject, "maximum_history_length")
    state_def = pycall(hyperparams.get, PyObject, "state")

    data_dir = normpath(joinpath(@__DIR__, "..",
                                "Trajectron-plus-plus", "experiments", "processed"));

    eval_env, eval_scene, online_scene, log =
        pycall(py"load_online_scene", NTuple{4, PyObject},
               data_dir, param.test_data_name, maximum_history_length, state_def,
               param.test_scene_id, param.incl_robot_node, param.start_time_idx,
               ado_id_removed)

    dto = eval_scene.dt;

    curr_time_idx = param.start_time_idx;
    max_time_idx = pybuiltin("int")(eval_scene.timesteps) - 1;
    @assert curr_time_idx < max_time_idx "curr_time_idx must be smaller than $(max_time_idx)!"

    if verbose
        for string in log
            println(string)
        end
        println("Looking at the $(param.test_data_name) sequence, data_id $(param.test_scene_id), start_idx $(curr_time_idx)");
    end
    return TrajectronSceneLoader(param, model_dir, eval_env, eval_scene, online_scene,
                                 state_def, dto, curr_time_idx, max_time_idx)
end

function fetch_ado_positions!(loader::TrajectronSceneLoader; return_full_state=false);
    input_dict = loader.eval_scene.get_clipped_input_dict(loader.curr_time_idx,
                                                          loader.state_def)
    input_dict_returned = Dict{PyObject, Vector{Float64}}();
    for key in keys(input_dict)
        if key != loader.online_scene.robot
            if return_full_state
                # full ado states
                input_dict_returned[key] = input_dict[key][:];
            else
                # positions only
                input_dict_returned[key] = [input_dict[key][1],
                                            input_dict[key][2]]
            end
        end
    end
    loader.curr_time_idx += 1; # increment curr_time_idx by 1.
    return input_dict_returned
end

function reduce_to_positions(input_dict::Dict{T, Vector{Float64}}) where T <: Union{PyObject, String}
    input_dict_returned = Dict{T, Vector{Float64}}();
    for key in keys(input_dict)
        # positions only
        input_dict_returned[key] = [input_dict[key][1],
                                    input_dict[key][2]]
    end
    return input_dict_returned
end

function convert_nodes_to_str(input_dict::Dict)
    if typeof(input_dict) == Dict{String, Vector{Float64}}
        return input_dict
    end
    input_dict_returned = Dict{String, Vector{Float64}}();
    for key in keys(input_dict)
        input_dict_returned[pybuiltin("str")(key)] = input_dict[key][:];
    end
    return input_dict_returned
end

function get_trajectory_for_ado(scene_loader::TrajectronSceneLoader,
                                init_time::Time,
                                ado_id::String,
                                sim_horizon::Float64)
    start_time_idx = scene_loader.param.start_time_idx;
    end_time_idx = start_time_idx + Int64(round(sim_horizon/scene_loader.dto, digits=5));

    ado_trajectory = Trajectory2D();

    time = init_time;
    for ii = start_time_idx:1:end_time_idx
        input_dict_py = scene_loader.eval_scene.get_clipped_input_dict(ii, scene_loader.state_def)
        input_dict = convert_nodes_to_str(input_dict_py)
        pos_dict = input_dict;
        for key in keys(pos_dict)
            pos_dict[key] = pos_dict[key][1:2];
        end
        if in(ado_id, keys(input_dict))
            ado_trajectory[time] = pos_dict[ado_id];
        end
        time += Duration(scene_loader.dto);
    end
    if isempty(ado_trajectory)
        @error(ArgumentError("$(ado_id) does not exist in the scenes!"))
    end
    if !all(diff(to_sec.(collect(keys(ado_trajectory)))) .≈ scene_loader.dto)
        @warn("Ado trajectory is discontinuous!")
    end
    return ado_trajectory
end

# Synthetic Scene Loader
struct SyntheticSceneParameter <: Parameter
    rng::AbstractRNG
end

mutable struct SyntheticSceneLoader <: SceneLoader
    param::SyntheticSceneParameter
    ado_pos_dict::Dict{String, Vector{Float64}}
end

function fetch_ado_positions!(loader::SyntheticSceneLoader,
                              prediction_dict::Dict{String, Array{Float64, 3}})
    @assert keys(loader.ado_pos_dict) == keys(prediction_dict)
    for key in keys(prediction_dict)
        # sample next position independently per ado agent from prediction_dict
        num_samples = size(prediction_dict[key], 1);
        sample_idx = rand(loader.param.rng, 1:num_samples)
        loader.ado_pos_dict[key] = prediction_dict[key][sample_idx, 1, :];
    end
    return deepcopy(loader.ado_pos_dict)
end

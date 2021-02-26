#///////////////////////////////////////
#// File Name: scene_loader_test.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/01/28
#// Description: Test code for src/scene_loader.jl
#///////////////////////////////////////

using Distributions
using LinearAlgebra
using PyCall
using Random
using RobotOS

@testset "Trajectron Scene Loader Test" begin
    conf_file_name = "config.json"
    test_data_name = "eth_test.pkl";
    test_scene_id = 0;
    start_time_idx = 2;
    incl_robot_node = false;

    param = TrajectronSceneParameter(conf_file_name, test_data_name,
                                     test_scene_id, start_time_idx,
                                     incl_robot_node);
    scene_loader = TrajectronSceneLoader(param, verbose=false);

    ado_positions = fetch_ado_positions!(scene_loader);

    ado_positions_converted = convert_nodes_to_str(ado_positions);
    @test typeof(collect(keys(ado_positions_converted))[1]) == String
    @test typeof(collect(values(ado_positions_converted))[1]) == Vector{Float64}
end

@testset "Trajectron Scene Loader Test (with Robot)" begin
    conf_file_name = "config.json"
    test_data_name = "eth_test.pkl";
    test_scene_id = 0;
    start_time_idx = 2;
    incl_robot_node = true;

    param = TrajectronSceneParameter(conf_file_name, test_data_name,
                                     test_scene_id, start_time_idx,
                                     incl_robot_node);
    scene_loader = TrajectronSceneLoader(param, verbose=false);
    online_nodes_list_py = scene_loader.online_scene.nodes
    online_nodes_list = []
    for node in online_nodes_list_py
        push!(online_nodes_list, pybuiltin("str")(node))
    end

    dto = 0.4;
    sim_horizon = 10.0;
    ado_trajectory = get_trajectory_for_ado(scene_loader, Time(0.0),
                                            "PEDESTRIAN/4", sim_horizon);

    @test to_sec(minimum(ado_trajectory)[1]) ≈ 2.0;
    @test all(maximum(ado_trajectory)[2] - minimum(ado_trajectory)[2] .≈ [13.25, 0.36]);
    @test to_sec(maximum(ado_trajectory)[1]) ≈ 7.2;

    ado_positions = fetch_ado_positions!(scene_loader);

    ado_positions_converted = convert_nodes_to_str(ado_positions);
    @test typeof(collect(keys(ado_positions_converted))[1]) == String
    @test typeof(collect(values(ado_positions_converted))[1]) == Vector{Float64}
    @test !in("PEDESTRIAN/ROBOT", keys(ado_positions_converted))

    scene_loader.curr_time_idx -= 1;
    ado_states_full = fetch_ado_positions!(scene_loader, return_full_state=true);
    ado_states_full_converted = convert_nodes_to_str(ado_states_full);
    @test keys(ado_positions_converted) == keys(ado_states_full_converted);
    for key in keys(ado_states_full_converted)
        @test length(ado_states_full_converted[key]) == 6;
        @test ado_states_full_converted[key][1:2] == ado_positions_converted[key];
    end

    scene_loader_removed = TrajectronSceneLoader(param, verbose=false,
                                                 ado_id_removed="PEDESTRIAN/1")
    online_nodes_list_removed_py = scene_loader_removed.online_scene.nodes
    online_nodes_list_removed = []
    for node in online_nodes_list_removed_py
        push!(online_nodes_list_removed, pybuiltin("str")(node))
    end
    @test setdiff(online_nodes_list, online_nodes_list_removed) == ["PEDESTRIAN/1"]
end

@testset "Synthetic Scene Loader Test" begin
    rng = MersenneTwister(123);

    param = SyntheticSceneParameter(rng);
    ado_pos_dict = Dict("Pedestrian/1" => [1.0, 1.0],
                        "Pedestrian/2" => [2.0, 1.0],
                        "Pedestrian/3" => [3.0, 1.0]);
    loader = SyntheticSceneLoader(param, ado_pos_dict);
    @test loader.ado_pos_dict == ado_pos_dict;

    num_samples = 100;
    dto = 0.4;
    prediction_steps = 12;
    deterministic = false;
    predictor_param = GaussianPredictorParameter(prediction_steps, num_samples, deterministic, rng);
    ado_vel_dict = Dict("Pedestrian/1" => MvNormal([0.0, 0.0], Diagonal([1.0, 1.0])),
                        "Pedestrian/2" => MvNormal([0.0, 0.0], Diagonal([1.0, 1.0])),
                        "Pedestrian/3" => MvNormal([0.0, 0.0], Diagonal([1.0, 1.0])));
    predictor = GaussianPredictor(predictor_param, dto, ado_vel_dict);

    prediction_dict = sample_future_ado_positions!(predictor, loader.ado_pos_dict);
    ado_positions = fetch_ado_positions!(loader, prediction_dict);
    @test typeof(ado_positions) == Dict{String, Vector{Float64}}
end

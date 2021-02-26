#///////////////////////////////////////
#// File Name: predictor_test.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/01/28
#// Description: Test code for src/predictor.jl
#///////////////////////////////////////

using Distributions
using LinearAlgebra
using PyCall
using Random

@testset "Trajectron Predictor Test" begin
    conf_file_name = "config.json"
    test_data_name = "eth_test.pkl";
    test_scene_id = 0;
    start_time_idx = 2;
    incl_robot_node = false;
    device = py"torch".device("cuda");

    scene_param = TrajectronSceneParameter(conf_file_name, test_data_name,
                                     test_scene_id, start_time_idx,
                                     incl_robot_node);
    prediction_steps = 12;
    use_robot_future = false;
    rng_seed_py = 12;
    deterministic = true;
    # Case 1: deterministic == true
    begin
        num_samples_1 = 1;
        rng_seed_py_1 = 1;
        scene_loader = TrajectronSceneLoader(scene_param, verbose=false);
        predictor_param_1 = TrajectronPredictorParameter(prediction_steps, num_samples_1,
                                                         use_robot_future, deterministic,
                                                         rng_seed_py_1);
        predictor_1 = TrajectronPredictor(predictor_param_1,
                                          scene_loader.model_dir,
                                          scene_loader.param.conf_file_name,
                                          device, verbose=false);
        initialize_scene_graph!(predictor_1, scene_loader)
        ado_states = fetch_ado_positions!(scene_loader, return_full_state=true);
        outputs_dict_1 = sample_future_ado_positions!(predictor_1, ado_states);
        @test size(collect(values(outputs_dict_1))[1]) == (num_samples_1, prediction_steps, 2);
        @test keys(convert_nodes_to_str(ado_states)) == keys(outputs_dict_1);
    end
    begin
        num_samples_2 = 1;
        rng_seed_py_2 = 123;
        scene_loader = TrajectronSceneLoader(scene_param, verbose=false);
        predictor_param_2 = TrajectronPredictorParameter(prediction_steps, num_samples_2,
                                                         use_robot_future, deterministic,
                                                         rng_seed_py_2);
        predictor_2 = TrajectronPredictor(predictor_param_2,
                                          scene_loader.model_dir,
                                          scene_loader.param.conf_file_name,
                                          device, verbose=false);
        initialize_scene_graph!(predictor_2, scene_loader)
        ado_states = fetch_ado_positions!(scene_loader, return_full_state=true);
        outputs_dict_2 = sample_future_ado_positions!(predictor_2, ado_states);
    end
    @test all(collect(values(outputs_dict_1))[1] .≈ collect(values(outputs_dict_2))[1])
    # Case 2: deterministic == false
    deterministic = false;
    begin
        num_samples_3 = 30;
        rng_seed_py_3 = 2;
        scene_loader = TrajectronSceneLoader(scene_param, verbose=false);
        predictor_param_3 = TrajectronPredictorParameter(prediction_steps, num_samples_3,
                                                         use_robot_future, deterministic,
                                                         rng_seed_py_3);
        predictor_3 = TrajectronPredictor(predictor_param_3,
                                          scene_loader.model_dir,
                                          scene_loader.param.conf_file_name,
                                          device, verbose=false);
        initialize_scene_graph!(predictor_3, scene_loader);
        ado_states = fetch_ado_positions!(scene_loader, return_full_state=true);
        outputs_dict_3 = sample_future_ado_positions!(predictor_3, ado_states);
        @test size(collect(values(outputs_dict_3))[1]) == (num_samples_3, prediction_steps, 2);
        @test keys(convert_nodes_to_str(ado_states)) == keys(outputs_dict_3);
    end
end

@testset "Trajectron Predictor Test (with Robot)" begin
    conf_file_name = "config.json"
    test_data_name = "eth_test.pkl";
    test_scene_id = 0;
    start_time_idx = 2;
    incl_robot_node = true;

    scene_param = TrajectronSceneParameter(conf_file_name, test_data_name,
                                           test_scene_id, start_time_idx,
                                           incl_robot_node);
    prediction_steps = 12;
    use_robot_future = true;
    rng_seed_py = 12;
    deterministic = true;
    device = py"torch".device("cpu");

    u_nominal_base = [0.0, 0.0];
    u_nominal_cand = append!([u_nominal_base],
                             [round.([a*cos(deg2rad(θ)), a*sin(deg2rad(θ))], digits=5)
                              for a = [2., 4.] for θ = 0.:45.:(360. - 45.)])

    ego_pos_init_vec = [5., 0.5];
    ego_pos_goal_vec = [5., 8.9];

    e_init = RobotState([ego_pos_init_vec; ego_pos_goal_vec]);
    robot_present_and_future = rand(length(u_nominal_cand), 1 + prediction_steps, 6);
    robot_present_and_future[1, :, :] .= 0.0;

    # Case 1: deterministic == true
    begin
        num_samples_1 = 1;
        rng_seed_py_1 = 1;
        scene_loader_1 = TrajectronSceneLoader(scene_param, verbose=false);
        predictor_param_1 = TrajectronPredictorParameter(prediction_steps, num_samples_1,
                                                         use_robot_future, deterministic,
                                                         rng_seed_py_1);
        predictor_1 = TrajectronPredictor(predictor_param_1,
                                          scene_loader_1.model_dir,
                                          scene_loader_1.param.conf_file_name,
                                          device, verbose=false);
        initialize_scene_graph!(predictor_1, scene_loader_1)
        ado_states_1 = fetch_ado_positions!(scene_loader_1, return_full_state=true);
        outputs_dict_1 = sample_future_ado_positions!(predictor_1, ado_states_1,
                                                      robot_present_and_future);
        @test size(collect(values(outputs_dict_1))[1]) == (length(u_nominal_cand), prediction_steps, 2);
        @test keys(convert_nodes_to_str(ado_states_1)) == keys(outputs_dict_1);
    end
    begin
        num_samples_2 = 1;
        rng_seed_py_2 = 123;
        scene_loader_2 = TrajectronSceneLoader(scene_param, verbose=false);
        predictor_param_2 = TrajectronPredictorParameter(prediction_steps, num_samples_2,
                                                         use_robot_future, deterministic,
                                                         rng_seed_py_2);
        predictor_2 = TrajectronPredictor(predictor_param_2,
                                          scene_loader_2.model_dir,
                                          scene_loader_2.param.conf_file_name,
                                          device, verbose=false);
        initialize_scene_graph!(predictor_2, scene_loader_2)
        ado_states_2 = fetch_ado_positions!(scene_loader_2, return_full_state=true);
        outputs_dict_2 = sample_future_ado_positions!(predictor_2, ado_states_2,
                                                      robot_present_and_future);
    end
    @test all(collect(values(outputs_dict_1))[1] .≈ collect(values(outputs_dict_2))[1])
    # Case 2: deterministic == false
    deterministic = false;
    begin
        num_samples_3 = 30;
        rng_seed_py_3 = 2;
        scene_loader_3 = TrajectronSceneLoader(scene_param, verbose=false);
        predictor_param_3 = TrajectronPredictorParameter(prediction_steps, num_samples_3,
                                                         use_robot_future, deterministic,
                                                         rng_seed_py_3);
        predictor_3 = TrajectronPredictor(predictor_param_3,
                                          scene_loader_3.model_dir,
                                          scene_loader_3.param.conf_file_name,
                                          device, verbose=false);
        initialize_scene_graph!(predictor_3, scene_loader_3);
        ado_states_3 = fetch_ado_positions!(scene_loader_3, return_full_state=true);
        outputs_dict_3 = sample_future_ado_positions!(predictor_3, ado_states_3,
                                                      robot_present_and_future);
        @test size(collect(values(outputs_dict_3))[1]) == (num_samples_3*length(u_nominal_cand), prediction_steps, 2);
        @test keys(convert_nodes_to_str(ado_states_3)) == keys(outputs_dict_3);
    end

    # Make sure permuted, reshaped outputs_dict in sample_future_ado_positions! is consistent with what's
    # required in forward-backward simulation. (num_samples*num_controls, prediction_steps, 2)
    begin
        dummy_array = rand(length(u_nominal_cand), num_samples_3,
                           prediction_steps, 2);
        squeezed_dummy_array = reshape(permutedims(dummy_array, [2, 1, 3, 4]), :,
                                       size(dummy_array, 3), size(dummy_array, 4));
        @test squeezed_dummy_array[1:num_samples_3, :, :] ==
                dummy_array[1, :, :, :];
        @test squeezed_dummy_array[1+num_samples_3:2*num_samples_3, :, :] ==
                dummy_array[2, :, :, :];
    end
end

@testset "Oracle Predictor Test" begin
    conf_file_name = "config.json"
    test_data_name = "eth_test.pkl";
    test_scene_id = 0;
    start_time_idx = 2;
    incl_robot_node = false;
    device = py"torch".device("cpu");

    scene_param = TrajectronSceneParameter(conf_file_name, test_data_name,
                                           test_scene_id, start_time_idx,
                                           incl_robot_node);
    prediction_steps = 50;
    scene_loader = TrajectronSceneLoader(scene_param, verbose=false);

    predictor_param = OraclePredictorParameter(prediction_steps);
    predictor = OraclePredictor(predictor_param, scene_loader);

    scene_loader.curr_time_idx = start_time_idx
    ado_pos_dict = fetch_ado_positions!(scene_loader)
    outputs_dict = sample_future_ado_positions!(predictor, ado_pos_dict);
    @test keys(convert_nodes_to_str(ado_pos_dict)) == keys(outputs_dict)

    function testfunc()
        for ii = 1:predictor.param.prediction_steps
            ado_pos_dict = fetch_ado_positions!(scene_loader)
            ado_pos_dict = convert_nodes_to_str(ado_pos_dict)
            for key in keys(outputs_dict)
                if in(key, keys(ado_pos_dict))
                    if !all(outputs_dict[key][1, ii, :] .≈ ado_pos_dict[key])
                        return false
                    end
                else
                    if !all(outputs_dict[key][1, ii, :] .== predictor.param.dummy_pos)
                        return false
                    end
                end
            end
        end
        return true;
    end
    @test testfunc()
end

@testset "Gaussian Predictor Test" begin
    rng = MersenneTwister(123);

    scene_param = SyntheticSceneParameter(rng);
    ado_pos_dict = Dict("Pedestrian/1" => [1.0, 1.0],
                        "Pedestrian/2" => [2.0, 1.0],
                        "Pedestrian/3" => [3.0, 1.0]);
    scene_loader = SyntheticSceneLoader(scene_param, ado_pos_dict);

    num_samples = 100;
    dto = 0.4;
    prediction_steps = 12;
    rng = MersenneTwister(123);
    # Case 1: deterministic == false
    deterministic = false
    begin
        predictor_param = GaussianPredictorParameter(prediction_steps, num_samples,
                                                     deterministic, rng);

        ado_vel_dict = Dict("Pedestrian/1" => MvNormal([0.0, 0.0], Diagonal([1.0, 1.0])),
                            "Pedestrian/2" => MvNormal([0.0, 0.0], Diagonal([1.0, 1.0])),
                            "Pedestrian/3" => MvNormal([0.0, 0.0], Diagonal([1.0, 1.0])));
        predictor = GaussianPredictor(predictor_param, dto, ado_vel_dict);
        outputs_dict = sample_future_ado_positions!(predictor, scene_loader.ado_pos_dict);
        @test size(collect(values(outputs_dict))[1]) == (num_samples, prediction_steps, 2);
        @test keys(outputs_dict) == keys(scene_loader.ado_pos_dict);

        ado_vel_dict_any = Dict("Any" => MvNormal([0.0, 0.0], Diagonal([1.0, 1.0])));
        predictor_any = GaussianPredictor(predictor_param, dto, ado_vel_dict_any);
        outputs_dict_any = sample_future_ado_positions!(predictor, scene_loader.ado_pos_dict);
        @test size(collect(values(outputs_dict_any))[1]) == (num_samples, prediction_steps, 2);
        @test keys(outputs_dict_any) == keys(scene_loader.ado_pos_dict);
    end
    # Case 2: deterministic == true
    deterministic = true
    begin
        rng_1 = MersenneTwister(1);
        num_samples = 1;
        predictor_param = GaussianPredictorParameter(prediction_steps, num_samples,
                                                     deterministic, rng_1);

        ado_vel_dict = Dict("Any" => MvNormal([1.0, 0.0], Diagonal([1.0, 1.0])));
        predictor = GaussianPredictor(predictor_param, dto, ado_vel_dict);
        outputs_dict_1 = sample_future_ado_positions!(predictor, scene_loader.ado_pos_dict);
        @test size(collect(values(outputs_dict_1))[1]) == (num_samples, prediction_steps, 2);
        @test keys(outputs_dict_1) == keys(scene_loader.ado_pos_dict);
        test_array = zeros(num_samples, prediction_steps, 2);
        for ii = 1:prediction_steps
            test_array[1, ii, 1] = 1.0 + 1.0*ii*dto;
            test_array[1, ii, 2] = 1.0;
        end
        @test all(collect(values(outputs_dict_1))[1] .≈ test_array)
    end
    begin
        rng_2 = MersenneTwister(123);
        num_samples = 1;
        predictor_param = GaussianPredictorParameter(prediction_steps, num_samples,
                                                     deterministic, rng_2);
        ado_vel_dict = Dict("Any" => MvNormal([1.0, 0.0], Diagonal([1.0, 1.0])));
        predictor = GaussianPredictor(predictor_param, dto, ado_vel_dict);
        outputs_dict_2 = sample_future_ado_positions!(predictor, scene_loader.ado_pos_dict);
    end
    all(collect(values(outputs_dict_1))[1] .≈ collect(values(outputs_dict_2))[1])
end

#///////////////////////////////////////
#// File Name: utils_test.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/02/25
#// Description: Test code for src/utils.jl
#///////////////////////////////////////

import StatsBase: fit
using Distributions
using LinearAlgebra
using PyCall
using Random
using RobotOS

# RSSAC Controller Setup Tests
@testset "Controller Setup: Synthetic + Gaussian" begin
    # Scene loader & Predictor parameters
    prediction_device = "cpu";
    prediction_steps = 12;
    ado_pos_init_dict = Dict("Pedestrian/1" => [0.0, -5.0]);
    ado_vel_dict = Dict("Pedestrian/1" => MvNormal([0.0, 1.0], Diagonal([0.01, 0.3])));
    dto = 0.4;
    deterministic = false;
    prediction_rng_seed = 1;
    num_samples = 50;
    # Cost Parameters
    Cep = Matrix(1.0I, 2, 2);
    Cu = Matrix(1.0I, 2, 2);
    β_pos = 0.6;
    β_col = 0.4;
    α_col = 40.0;
    λ_col = 2.0;
    σ_risk = 0.0;
    # Control Parameters
    dtc = 0.01;
    dtr = 0.1;
    dtexec = [0.02, 0.06];
    tcalc = 0.1;
    u_norm_max = 5.0;
    u_nominal_base = [0.0, 0.0];
    u_nominal_cand = append!([u_nominal_base],
                             [round.([a*cos(deg2rad(θ)), a*sin(deg2rad(θ))], digits=5)
                              for a = [2., 4.] for θ = 0.:45.:(360. - 45.)])            # nominal control candidate value [ax, ay] [m/s^2]
    nominal_search_depth = 1;
    # Ego initial state
    ego_pos_init_vec = [-5., 0.];
    ego_pos_goal_vec = [5., 0.];
    # Other parameters
    target_speed = 1.0;
    sim_horizon = 10.0;


    rng = MersenneTwister(prediction_rng_seed);
    scene_param = SyntheticSceneParameter(rng);
    predictor_param = GaussianPredictorParameter(prediction_steps,
                                                 num_samples, deterministic, rng);
    cost_param = CostParameter(Cep, Cu, β_pos, α_col, β_col, λ_col, σ_risk);
    cnt_param = ControlParameter(u_norm_max, tcalc, dtexec, dtr,
                                 u_nominal_base, u_nominal_cand, nominal_search_depth);

    scene_loader, controller, w_init, measurement_schedule, target_trajectory, ~ =
        controller_setup(scene_param,
                         predictor_param,
                         prediction_device=prediction_device,
                         cost_param=cost_param,
                         cnt_param=cnt_param,
                         dtc=dtc,
                         dto=dto,
                         ado_pos_init_dict=ado_pos_init_dict,
                         ado_vel_dict=ado_vel_dict,
                         ego_pos_init_vec=ego_pos_init_vec,
                         ego_vel_init_vec=nothing,
                         ego_pos_goal_vec=ego_pos_goal_vec,
                         target_speed=target_speed,
                         sim_horizon=sim_horizon,
                         verbose=false)

    @test scene_loader.param.rng === controller.predictor.param.rng
    @test scene_loader.ado_pos_dict == ado_pos_init_dict
    @test controller.sim_param.dtc == dtc;
    @test controller.sim_param.dto == dto;
    @test controller.sim_param.prediction_steps == prediction_steps;
    @test controller.sim_param.num_samples == num_samples;
    @test controller.sim_param.cost_param.Cep == Cep;
    @test controller.sim_param.cost_param.Cu == Cu;
    @test controller.sim_param.cost_param.β_pos == β_pos;
    @test controller.sim_param.cost_param.α_col == α_col;
    @test controller.sim_param.cost_param.β_col == β_col;
    @test controller.sim_param.cost_param.λ_col == λ_col;
    @test controller.sim_param.cost_param.σ_risk == σ_risk;
    @test controller.cnt_param.eamax == u_norm_max;
    @test controller.cnt_param.tcalc == tcalc;
    @test controller.cnt_param.dtexec == dtexec;
    @test controller.cnt_param.dtr == dtr;
    @test controller.predictor.dto == dto;
    @test controller.predictor.param.deterministic == deterministic;
    @test controller.predictor.param.prediction_steps == prediction_steps;
    @test controller.predictor.param.num_samples == num_samples;
    @test controller.predictor.ado_vel_dict == ado_vel_dict;
    @test measurement_schedule[1] == Time(controller.sim_param.dto);
    @test measurement_schedule[end] == Time(sim_horizon);
    @test minimum(target_trajectory)[1] == Time(0.0);
    @test minimum(target_trajectory)[2] == ego_pos_init_vec;
    @test maximum(target_trajectory)[1] == Time(14.8);
    @test maximum(target_trajectory)[2] == ego_pos_goal_vec;
    @test get_position(w_init.e_state) == ego_pos_init_vec;
    @test get_velocity(w_init.e_state) ==
        (ego_pos_goal_vec - ego_pos_init_vec)./
        norm(ego_pos_goal_vec - ego_pos_init_vec)*target_speed;
end

@testset "Controller Setup: Data + Gaussian" begin
    # Scene loader & Predictor parameters
    conf_file_name = "config.json";
    test_data_name = "eth_test.pkl";
    test_scene_id = 0;
    start_time_idx = 3;
    incl_robot_node = false;
    prediction_device = "cpu";
    prediction_steps = 12;
    ado_vel_dict = Dict("Any" => MvNormal([0.0, 0.0], Diagonal([0.8, 0.8])));
    deterministic = false;
    prediction_rng_seed = 1;
    num_samples = 50;
    # Cost Parameters
    Cep = Matrix(1.0I, 2, 2);
    Cu = Matrix(1.0I, 2, 2);
    β_pos = 0.6;
    β_col = 0.4;
    α_col = 40.0;
    λ_col = 2.0;
    σ_risk = 0.0;
    # Control Parameters
    dtc = 0.01;
    dtr = 0.1;
    dtexec = [0.02, 0.06];;
    tcalc = 0.1;
    u_norm_max = 5.0;
    u_nominal_base = [0.0, 0.0];
    u_nominal_cand = append!([u_nominal_base],
                             [round.([a*cos(deg2rad(θ)), a*sin(deg2rad(θ))], digits=5)
                              for a = [2., 4.] for θ = 0.:45.:(360. - 45.)])            # nominal control candidate value [ax, ay] [m/s^2]
    nominal_search_depth = 1;
    # Ego initial state
    ego_pos_init_vec = [-5., 0.];
    ego_pos_goal_vec = [5., 0.];
    # Other parameters
    target_speed = 1.0;
    sim_horizon = 10.0;


    scene_param = TrajectronSceneParameter(conf_file_name, test_data_name,
                                           test_scene_id, start_time_idx,
                                           incl_robot_node);
    rng = MersenneTwister(prediction_rng_seed);
    predictor_param = GaussianPredictorParameter(prediction_steps,
                                                 num_samples, deterministic, rng);
    cost_param = CostParameter(Cep, Cu, β_pos, α_col, β_col, λ_col, σ_risk);
    cnt_param = ControlParameter(u_norm_max, tcalc, dtexec, dtr,
                                 u_nominal_base, u_nominal_cand, nominal_search_depth);

    scene_loader, controller, w_init, measurement_schedule, target_trajectory, ~ =
    controller_setup(scene_param,
                     predictor_param,
                     prediction_device=prediction_device,
                     cost_param=cost_param,
                     cnt_param=cnt_param,
                     dtc=dtc,
                     ado_vel_dict=ado_vel_dict,
                     ego_pos_init_vec=ego_pos_init_vec,
                     ego_vel_init_vec=nothing,
                     ego_pos_goal_vec=ego_pos_goal_vec,
                     target_speed=target_speed,
                     sim_horizon=sim_horizon,
                     verbose=false)

    @test scene_loader.param.conf_file_name == conf_file_name;
    @test scene_loader.param.test_data_name == test_data_name;
    @test scene_loader.param.test_scene_id == test_scene_id;
    @test scene_loader.param.start_time_idx == start_time_idx;
    @test scene_loader.param.incl_robot_node == false;
    @test scene_loader.curr_time_idx == start_time_idx + 1;
    @test controller.sim_param.dtc == dtc;
    @test controller.sim_param.dto == scene_loader.dto;
    @test controller.sim_param.prediction_steps == prediction_steps;
    @test controller.sim_param.num_samples == num_samples;
    @test controller.sim_param.cost_param.Cep == Cep;
    @test controller.sim_param.cost_param.Cu == Cu;
    @test controller.sim_param.cost_param.β_pos == β_pos;
    @test controller.sim_param.cost_param.α_col == α_col;
    @test controller.sim_param.cost_param.β_col == β_col;
    @test controller.sim_param.cost_param.λ_col == λ_col;
    @test controller.sim_param.cost_param.σ_risk == σ_risk;
    @test controller.cnt_param.eamax == u_norm_max;
    @test controller.cnt_param.tcalc == tcalc;
    @test controller.cnt_param.dtexec == dtexec;
    @test controller.cnt_param.dtr == dtr;
    @test controller.predictor.dto == scene_loader.dto;
    @test controller.predictor.param.deterministic == deterministic;
    @test controller.predictor.param.prediction_steps == prediction_steps;
    @test controller.predictor.param.num_samples == num_samples;
    @test controller.predictor.ado_vel_dict == ado_vel_dict;
    @test measurement_schedule[1] == Time(controller.sim_param.dto);
    @test measurement_schedule[end] == Time(sim_horizon);
    @test minimum(target_trajectory)[1] == Time(0.0);
    @test minimum(target_trajectory)[2] == ego_pos_init_vec;
    @test maximum(target_trajectory)[1] == Time(14.8);
    @test maximum(target_trajectory)[2] == ego_pos_goal_vec;
    @test get_position(w_init.e_state) == ego_pos_init_vec;
    @test get_velocity(w_init.e_state) ==
        (ego_pos_goal_vec - ego_pos_init_vec)./
        norm(ego_pos_goal_vec - ego_pos_init_vec)*target_speed;
end

@testset "Controller Setup: Data + Trajectron" begin
    # Scene loader & Predictor parameters
    conf_file_name = "config.json";
    test_data_name = "eth_test.pkl";
    test_scene_id = 0;
    start_time_idx = 2;
    incl_robot_node = false;
    prediction_device = "cpu";
    deterministic = false;
    prediction_rng_seed = 1;
    prediction_steps = 12;
    num_samples = 50;
    use_robot_future = false;
    # Cost Parameters
    Cep = Matrix(1.0I, 2, 2);
    Cu = Matrix(1.0I, 2, 2);
    β_pos = 0.6;
    β_col = 0.4;
    α_col = 40.0;
    λ_col = 2.0;
    σ_risk = 0.0;
    # Control Parameters
    dtc = 0.01;
    dtr = 0.1;
    dtexec = [0.02, 0.06];;
    tcalc = 0.1;
    u_norm_max = 5.0;
    u_nominal_base = [0.0, 0.0];
    u_nominal_cand = append!([u_nominal_base],
                             [round.([a*cos(deg2rad(θ)), a*sin(deg2rad(θ))], digits=5)
                              for a = [2., 4.] for θ = 0.:45.:(360. - 45.)])            # nominal control candidate value [ax, ay] [m/s^2]
    nominal_search_depth = 1;
    # Ego initial state
    ego_pos_init_vec = [5.0, 0.5];
    ego_pos_goal_vec = [5.0, 7.5];
    # Other parameters
    target_speed = 0.7;
    sim_horizon = 10.0;


    scene_param = TrajectronSceneParameter(conf_file_name, test_data_name,
                                           test_scene_id, start_time_idx,
                                           incl_robot_node);
    predictor_param = TrajectronPredictorParameter(prediction_steps,
                                                   num_samples,
                                                   use_robot_future,
                                                   deterministic,
                                                   prediction_rng_seed);
    cost_param = CostParameter(Cep, Cu, β_pos, α_col, β_col, λ_col, σ_risk);
    cnt_param = ControlParameter(u_norm_max, tcalc, dtexec, dtr,
                                 u_nominal_base, u_nominal_cand, nominal_search_depth);
    scene_loader, controller, w_init, measurement_schedule, target_trajectory, ~ =
    controller_setup(scene_param,
                     predictor_param,
                     prediction_device=prediction_device,
                     cost_param=cost_param,
                     cnt_param=cnt_param,
                     dtc=dtc,
                     ego_pos_init_vec=ego_pos_init_vec,
                     ego_vel_init_vec=nothing,
                     ego_pos_goal_vec=ego_pos_goal_vec,
                     target_speed=target_speed,
                     sim_horizon=sim_horizon,
                     verbose=false)

    @test scene_loader.param.conf_file_name == conf_file_name;
    @test scene_loader.param.test_data_name == test_data_name;
    @test scene_loader.param.test_scene_id == test_scene_id;
    @test scene_loader.param.start_time_idx == start_time_idx;
    @test scene_loader.param.incl_robot_node == false;
    @test scene_loader.curr_time_idx == start_time_idx + 1;
    @test controller.sim_param.dtc == dtc;
    @test controller.sim_param.dto == scene_loader.dto;
    @test controller.sim_param.prediction_steps == prediction_steps;
    @test controller.sim_param.num_samples == num_samples;
    @test controller.sim_param.cost_param.Cep == Cep;
    @test controller.sim_param.cost_param.Cu == Cu;
    @test controller.sim_param.cost_param.β_pos == β_pos;
    @test controller.sim_param.cost_param.α_col == α_col;
    @test controller.sim_param.cost_param.β_col == β_col;
    @test controller.sim_param.cost_param.λ_col == λ_col;
    @test controller.sim_param.cost_param.σ_risk == σ_risk;
    @test controller.cnt_param.eamax == u_norm_max;
    @test controller.cnt_param.tcalc == tcalc;
    @test controller.cnt_param.dtexec == dtexec;
    @test controller.cnt_param.dtr == dtr;
    @test controller.predictor.param.deterministic == deterministic;
    @test controller.predictor.param.prediction_steps == prediction_steps;
    @test controller.predictor.param.num_samples == num_samples;
    @test controller.predictor.param.use_robot_future == false;
    @test controller.predictor.param.deterministic == false;
    @test controller.predictor.param.rng_seed_py == prediction_rng_seed;
    @test istaskdone(controller.prediction_task);
    @test !isnothing(controller.prediction_dict_tmp);
    @test w_init.t == Time(0.0);
    @test keys(w_init.ap_dict) == keys(controller.prediction_dict_tmp);
    @test w_init.t_last_m == Time(0.0);
    @test measurement_schedule[1] == Time(controller.sim_param.dto);
    @test measurement_schedule[end] == Time(sim_horizon);
    @test minimum(target_trajectory)[1] == Time(0.0);
    @test minimum(target_trajectory)[2] == ego_pos_init_vec;
    @test maximum(target_trajectory)[1] == Time(14.8);
    @test maximum(target_trajectory)[2] == ego_pos_goal_vec;
    @test get_position(w_init.e_state) == ego_pos_init_vec;
    @test get_velocity(w_init.e_state) ==
        (ego_pos_goal_vec - ego_pos_init_vec)./
        norm(ego_pos_goal_vec - ego_pos_init_vec)*target_speed;
end

@testset "Controller Setup: Data (Ado Replaced) + Trajectron" begin
    # Scene loader & Predictor parameters
    conf_file_name = "config.json";
    test_data_name = "eth_test.pkl";
    test_scene_id = 0;
    start_time_idx = 2;
    incl_robot_node = false;
    prediction_device = "cpu";
    deterministic = false;
    prediction_rng_seed = 1;
    prediction_steps = 12;
    num_samples = 50;
    use_robot_future = false;
    # Cost Parameters
    Cep = Matrix(1.0I, 2, 2);
    Cu = Matrix(1.0I, 2, 2);
    β_pos = 0.6;
    β_col = 0.4;
    α_col = 40.0;
    λ_col = 2.0;
    σ_risk = 0.0;
    # Control Parameters
    dtc = 0.01;
    dtr = 0.1;
    dtexec = [0.02, 0.06];;
    tcalc = 0.1;
    u_norm_max = 5.0;
    u_nominal_base = [0.0, 0.0];
    u_nominal_cand = append!([u_nominal_base],
                             [round.([a*cos(deg2rad(θ)), a*sin(deg2rad(θ))], digits=5)
                              for a = [2., 4.] for θ = 0.:45.:(360. - 45.)])            # nominal control candidate value [ax, ay] [m/s^2]
    nominal_search_depth = 1;
    # Other parameters
    sim_horizon = 10.0;


    scene_param = TrajectronSceneParameter(conf_file_name, test_data_name,
                                           test_scene_id, start_time_idx,
                                           incl_robot_node);
    predictor_param = TrajectronPredictorParameter(prediction_steps,
                                                   num_samples,
                                                   use_robot_future,
                                                   deterministic,
                                                   prediction_rng_seed);
    cost_param = CostParameter(Cep, Cu, β_pos, α_col, β_col, λ_col, σ_risk);
    cnt_param = ControlParameter(u_norm_max, tcalc, dtexec, dtr,
                                 u_nominal_base, u_nominal_cand, nominal_search_depth);
    scene_loader, controller, w_init, measurement_schedule, target_trajectory, target_speed =
    controller_setup(scene_param,
                     predictor_param,
                     prediction_device=prediction_device,
                     cost_param=cost_param,
                     cnt_param=cnt_param,
                     dtc=dtc,
                     sim_horizon=sim_horizon,
                     ado_id_to_replace="PEDESTRIAN/2",
                     verbose=false)

    @test scene_loader.param.conf_file_name == conf_file_name;
    @test scene_loader.param.test_data_name == test_data_name;
    @test scene_loader.param.test_scene_id == test_scene_id;
    @test scene_loader.param.start_time_idx == start_time_idx;
    @test scene_loader.param.incl_robot_node == false;
    @test scene_loader.curr_time_idx == start_time_idx + 1;
    @test controller.sim_param.dtc == dtc;
    @test controller.sim_param.dto == scene_loader.dto;
    @test controller.sim_param.prediction_steps == prediction_steps;
    @test controller.sim_param.num_samples == num_samples;
    @test controller.sim_param.cost_param.Cep == Cep;
    @test controller.sim_param.cost_param.Cu == Cu;
    @test controller.sim_param.cost_param.β_pos == β_pos;
    @test controller.sim_param.cost_param.α_col == α_col;
    @test controller.sim_param.cost_param.β_col == β_col;
    @test controller.sim_param.cost_param.λ_col == λ_col;
    @test controller.sim_param.cost_param.σ_risk == σ_risk;
    @test controller.cnt_param.eamax == u_norm_max;
    @test controller.cnt_param.tcalc == tcalc;
    @test controller.cnt_param.dtexec == dtexec;
    @test controller.cnt_param.dtr == dtr;
    @test controller.predictor.param.deterministic == deterministic;
    @test controller.predictor.param.prediction_steps == prediction_steps;
    @test controller.predictor.param.num_samples == num_samples;
    @test controller.predictor.param.use_robot_future == false;
    @test controller.predictor.param.deterministic == false;
    @test controller.predictor.param.rng_seed_py == prediction_rng_seed;
    @test istaskdone(controller.prediction_task);
    @test !isnothing(controller.prediction_dict_tmp);
    @test w_init.t == Time(0.0);
    @test !in("PEDESTRIAN/2", keys(w_init.ap_dict))
    @test keys(w_init.ap_dict) == keys(controller.prediction_dict_tmp);
    @test w_init.t_last_m == Time(0.0);
    @test measurement_schedule[1] == Time(controller.sim_param.dto);
    @test measurement_schedule[end] == Time(sim_horizon);
    @test minimum(target_trajectory)[1] == Time(0.0);
    @test maximum(target_trajectory)[1] == Time(14.8);
    @test minimum(target_trajectory)[2] == get_position(w_init.e_state);
    @test all(get_velocity(w_init.e_state) .≈
              (collect(values(target_trajectory))[2] - minimum(target_trajectory)[2])./scene_loader.dto)
    @test target_speed ≈ norm((maximum(target_trajectory)[2] - minimum(target_trajectory)[2])./
                               to_sec(collect(keys(target_trajectory))[end-1] - minimum(target_trajectory)[1]))
end

@testset "Parameter Setup: Data + Oracle" begin
    # Scene loader & Predictor parameters
    conf_file_name = "config.json";
    test_data_name = "eth_test.pkl";
    test_scene_id = 0;
    start_time_idx = 3;
    incl_robot_node = false;
    prediction_device = "cpu";
    deterministic = true;
    prediction_steps = 12;
    num_samples = 1;
    # Cost Parameters
    Cep = Matrix(1.0I, 2, 2);
    Cu = Matrix(1.0I, 2, 2);
    β_pos = 0.6;
    β_col = 0.4;
    α_col = 40.0;
    λ_col = 2.0;
    σ_risk = 0.0;
    # Control Parameters
    dtc = 0.01;
    dtr = 0.1;
    dtexec = [0.02, 0.06];;
    tcalc = 0.1;
    u_norm_max = 5.0;
    u_nominal_base = [0.0, 0.0];
    u_nominal_cand = append!([u_nominal_base],
                             [round.([a*cos(deg2rad(θ)), a*sin(deg2rad(θ))], digits=5)
                              for a = [2., 4.] for θ = 0.:45.:(360. - 45.)])            # nominal control candidate value [ax, ay] [m/s^2]
    nominal_search_depth = 1;
    # Ego initial state
    ego_pos_init_vec = [5.0, 0.5];
    ego_pos_goal_vec = [5.0, 7.5];
    # Other parameters
    target_speed = 0.7;
    sim_horizon = 10.0;


    scene_param = TrajectronSceneParameter(conf_file_name, test_data_name,
                                           test_scene_id, start_time_idx,
                                           incl_robot_node);
    predictor_param = OraclePredictorParameter(prediction_steps);
    cost_param = CostParameter(Cep, Cu, β_pos, α_col, β_col, λ_col, σ_risk);
    cnt_param = ControlParameter(u_norm_max, tcalc, dtexec, dtr,
                                 u_nominal_base, u_nominal_cand, nominal_search_depth);

    scene_loader, controller, w_init, measurement_schedule, target_trajectory, ~ =
    controller_setup(scene_param,
                     predictor_param,
                     prediction_device=prediction_device,
                     cost_param=cost_param,
                     cnt_param=cnt_param,
                     dtc=dtc,
                     ego_pos_init_vec=ego_pos_init_vec,
                     ego_vel_init_vec=nothing,
                     ego_pos_goal_vec=ego_pos_goal_vec,
                     target_speed=target_speed,
                     sim_horizon=sim_horizon,
                     verbose=false)

    @test scene_loader.param.conf_file_name == conf_file_name;
    @test scene_loader.param.test_data_name == test_data_name;
    @test scene_loader.param.test_scene_id == test_scene_id;
    @test scene_loader.param.start_time_idx == start_time_idx;
    @test scene_loader.param.incl_robot_node == false;
    @test scene_loader.curr_time_idx == start_time_idx + 1;
    @test controller.sim_param.dtc == dtc;
    @test controller.sim_param.dto == scene_loader.dto;
    @test controller.sim_param.prediction_steps == prediction_steps;
    @test controller.sim_param.num_samples == num_samples;
    @test controller.sim_param.cost_param.Cep == Cep;
    @test controller.sim_param.cost_param.Cu == Cu;
    @test controller.sim_param.cost_param.β_pos == β_pos;
    @test controller.sim_param.cost_param.α_col == α_col;
    @test controller.sim_param.cost_param.β_col == β_col;
    @test controller.sim_param.cost_param.λ_col == λ_col;
    @test controller.sim_param.cost_param.σ_risk == σ_risk;
    @test controller.cnt_param.eamax == u_norm_max;
    @test controller.cnt_param.tcalc == tcalc;
    @test controller.cnt_param.dtexec == dtexec;
    @test controller.cnt_param.dtr == dtr;
    @test controller.predictor.param.prediction_steps == prediction_steps;
    @test controller.predictor.eval_scene == scene_loader.eval_scene;
    @test controller.predictor.state_def == scene_loader.state_def;
    @test controller.predictor.curr_time_idx == scene_loader.curr_time_idx
    @test controller.predictor.max_time_idx == scene_loader.max_time_idx
    @test istaskdone(controller.prediction_task);
    @test !isnothing(controller.prediction_dict_tmp);
    @test w_init.t == Time(0.0);
    @test keys(w_init.ap_dict) == keys(controller.prediction_dict_tmp);
    @test w_init.t_last_m == Time(0.0);
    @test measurement_schedule[1] == Time(controller.sim_param.dto);
    @test measurement_schedule[end] == Time(sim_horizon);
    @test minimum(target_trajectory)[1] == Time(0.0);
    @test minimum(target_trajectory)[2] == ego_pos_init_vec;
    @test maximum(target_trajectory)[1] == Time(14.8);
    @test maximum(target_trajectory)[2] == ego_pos_goal_vec;
    @test get_position(w_init.e_state) == ego_pos_init_vec;
    @test get_velocity(w_init.e_state) ==
        (ego_pos_goal_vec - ego_pos_init_vec)./
        norm(ego_pos_goal_vec - ego_pos_init_vec)*target_speed;
end

# BICController Setup Tests
@testset "Controller Setup: Synthetic + BIC" begin
    # Scene loader & Predictor parameters
    prediction_steps = 12;
    ado_pos_init_dict = Dict("Pedestrian/1" => [0.0, -5.0]);
    ado_vel_dict = Dict("Pedestrian/1" => MvNormal([0.0, 1.0], Diagonal([0.01, 0.3])));
    dto = 0.4;
    sim_rng_seed = 1;
    num_samples = 50;
    # Cost Parameters
    Cep = Matrix(1.0I, 2, 2);
    Cu = Matrix(1.0I, 2, 2);
    β_pos = 0.6;
    β_col = 0.4;
    α_col = 40.0;
    λ_col = 2.0;
    σ_risk = 0.0;
    # Control Parameters
    dtc = 0.01;
    dtr = 0.1;
    tcalc = 0.1;
    u_norm_max = 5.0;
    min_dist = 0.5;
    # Ego initial state
    ego_pos_init_vec = [-5., 0.];
    ego_pos_goal_vec = [5., 0.];
    # Other parameters
    target_speed = 1.0;
    sim_horizon = 10.0;


    rng = MersenneTwister(sim_rng_seed);
    scene_param = SyntheticSceneParameter(rng);
    cost_param = CostParameter(Cep, Cu, β_pos, α_col, β_col, λ_col, σ_risk);
    cnt_param = BICControlParameter(u_norm_max, tcalc, dtr, min_dist);

    scene_loader, controller, w_init, measurement_schedule, target_trajectory, ~ =
    controller_setup(scene_param,
                     cost_param=cost_param,
                     cnt_param=cnt_param,
                     dtc=dtc,
                     dto=dto,
                     prediction_steps=prediction_steps,
                     num_samples=num_samples,
                     sim_rng=rng,
                     ado_pos_init_dict=ado_pos_init_dict,
                     ado_vel_dict=ado_vel_dict,
                     ego_pos_init_vec=ego_pos_init_vec,
                     ego_vel_init_vec=nothing,
                     ego_pos_goal_vec=ego_pos_goal_vec,
                     target_speed=target_speed,
                     sim_horizon=sim_horizon,
                     verbose=false)

    @test scene_loader.ado_pos_dict == ado_pos_init_dict
    @test controller.sim_param.dtc == dtc;
    @test controller.sim_param.dto == dto;
    @test controller.sim_param.prediction_steps == prediction_steps;
    @test controller.sim_param.num_samples == num_samples;
    @test controller.sim_param.cost_param.Cep == Cep;
    @test controller.sim_param.cost_param.Cu == Cu;
    @test controller.sim_param.cost_param.β_pos == β_pos;
    @test controller.sim_param.cost_param.α_col == α_col;
    @test controller.sim_param.cost_param.β_col == β_col;
    @test controller.sim_param.cost_param.λ_col == λ_col;
    @test controller.sim_param.cost_param.σ_risk == σ_risk;
    @test controller.cnt_param.eamax == u_norm_max;
    @test controller.cnt_param.tcalc == tcalc;
    @test controller.cnt_param.dtr == dtr;
    @test controller.cnt_param.min_dist == min_dist;
    @test measurement_schedule[1] == Time(controller.sim_param.dto);
    @test measurement_schedule[end] == Time(sim_horizon);
    @test minimum(target_trajectory)[1] == Time(0.0);
    @test minimum(target_trajectory)[2] == ego_pos_init_vec;
    @test maximum(target_trajectory)[1] == Time(14.8);
    @test maximum(target_trajectory)[2] == ego_pos_goal_vec;
    @test get_position(w_init.e_state) == ego_pos_init_vec;
    @test get_velocity(w_init.e_state) ==
        (ego_pos_goal_vec - ego_pos_init_vec)./
        norm(ego_pos_goal_vec - ego_pos_init_vec)*target_speed;
end

@testset "Controller Setup: Data + BIC" begin
    # Scene loader & Predictor parameters
    conf_file_name = "config.json";
    test_data_name = "eth_test.pkl";
    test_scene_id = 0;
    start_time_idx = 3;
    incl_robot_node = false;
    prediction_steps = 12;
    # Cost Parameters
    Cep = Matrix(1.0I, 2, 2);
    Cu = Matrix(1.0I, 2, 2);
    β_pos = 0.6;
    β_col = 0.4;
    α_col = 40.0;
    λ_col = 2.0;
    σ_risk = 0.0;
    # Control Parameters
    dtc = 0.01;
    dtr = 0.1;
    tcalc = 0.1;
    u_norm_max = 5.0;
    min_dist = 0.5;
    # Ego initial state
    ego_pos_init_vec = [-5., 0.];
    ego_pos_goal_vec = [5., 0.];
    # Other parameters
    target_speed = 1.0;
    sim_horizon = 10.0;


    scene_param = TrajectronSceneParameter(conf_file_name, test_data_name,
                                           test_scene_id, start_time_idx,
                                           incl_robot_node);
    cost_param = CostParameter(Cep, Cu, β_pos, α_col, β_col, λ_col, σ_risk);
    cnt_param = BICControlParameter(u_norm_max, tcalc, dtr, min_dist);

    scene_loader, controller, w_init, measurement_schedule, target_trajectory, ~ =
    controller_setup(scene_param,
                     cost_param=cost_param,
                     cnt_param=cnt_param,
                     dtc=dtc,
                     prediction_steps=prediction_steps,
                     ego_pos_init_vec=ego_pos_init_vec,
                     ego_pos_goal_vec=ego_pos_goal_vec,
                     target_speed=target_speed,
                     sim_horizon=sim_horizon,
                     verbose=false);

    @test scene_loader.param.conf_file_name == conf_file_name;
    @test scene_loader.param.test_data_name == test_data_name;
    @test scene_loader.param.test_scene_id == test_scene_id;
    @test scene_loader.param.start_time_idx == start_time_idx;
    @test scene_loader.param.incl_robot_node == false;
    @test scene_loader.curr_time_idx == start_time_idx + 1;
    @test controller.sim_param.dtc == dtc;
    @test controller.sim_param.dto == scene_loader.dto;
    @test controller.sim_param.prediction_steps == prediction_steps;
    @test controller.sim_param.num_samples == 0;
    @test controller.sim_param.cost_param.Cep == Cep;
    @test controller.sim_param.cost_param.Cu == Cu;
    @test controller.sim_param.cost_param.β_pos == β_pos;
    @test controller.sim_param.cost_param.α_col == α_col;
    @test controller.sim_param.cost_param.β_col == β_col;
    @test controller.sim_param.cost_param.λ_col == λ_col;
    @test controller.sim_param.cost_param.σ_risk == σ_risk;
    @test controller.cnt_param.eamax == u_norm_max;
    @test controller.cnt_param.tcalc == tcalc;
    @test controller.cnt_param.dtr == dtr;
    @test controller.cnt_param.min_dist == min_dist;
    @test measurement_schedule[1] == Time(controller.sim_param.dto);
    @test measurement_schedule[end] == Time(sim_horizon);
    @test minimum(target_trajectory)[1] == Time(0.0);
    @test minimum(target_trajectory)[2] == ego_pos_init_vec;
    @test maximum(target_trajectory)[1] == Time(14.8);
    @test maximum(target_trajectory)[2] == ego_pos_goal_vec;
    @test get_position(w_init.e_state) == ego_pos_init_vec;
    @test get_velocity(w_init.e_state) ==
        (ego_pos_goal_vec - ego_pos_init_vec)./
        norm(ego_pos_goal_vec - ego_pos_init_vec)*target_speed;
end

@testset "Controller Setup: Data + CrowdNav" begin
    # Scene loader & Predictor parameters
    conf_file_name = "config.json";
    test_data_name = "eth_test.pkl";
    test_scene_id = 0;
    start_time_idx = 3;
    incl_robot_node = false;
    prediction_steps = 12;
    # Cost Parameters
    Cep = Matrix(1.0I, 2, 2);
    Cu = Matrix(1.0I, 2, 2);
    β_pos = 0.6;
    β_col = 0.4;
    α_col = 40.0;
    λ_col = 2.0;
    σ_risk = 0.0;
    # Control Parameters
    model_dir = normpath(joinpath(@__DIR__,
                                  "../CrowdNav/crowd_nav/data/output_om_sarl_radius_0.4"));
    env_config = "env.config";
    policy_config = "policy.config";
    policy_name = "sarl";
    tcalc = 0.01;
    dtr = 0.4;
    # Ego initial state
    ego_pos_init_vec = [-5., 0.];
    ego_pos_goal_vec = [5., 0.];
    # Other parameters
    dtc = 0.01;
    target_speed = 1.0;
    sim_horizon = 10.0;

    scene_param = TrajectronSceneParameter(conf_file_name, test_data_name,
                                           test_scene_id, start_time_idx,
                                           incl_robot_node);
    cost_param = CostParameter(Cep, Cu, β_pos, α_col, β_col, λ_col, σ_risk);
    cnt_param = CrowdNavControlParameter(model_dir, env_config, policy_config,
                                         policy_name, ego_pos_goal_vec,
                                         tcalc, dtr);

    scene_loader, controller, w_init, ado_inputs, measurement_schedule,
    target_trajectory, target_speed =
    controller_setup(scene_param, cnt_param,
                     cost_param=cost_param,
                     dtc=dtc,
                     prediction_steps=prediction_steps,
                     ego_pos_init_vec=ego_pos_init_vec,
                     ego_pos_goal_vec=ego_pos_goal_vec,
                     target_speed=target_speed,
                     sim_horizon=sim_horizon,
                     verbose=false);
    @test scene_loader.param.conf_file_name == conf_file_name;
    @test scene_loader.param.test_data_name == test_data_name;
    @test scene_loader.param.test_scene_id == test_scene_id;
    @test scene_loader.param.start_time_idx == start_time_idx;
    @test scene_loader.param.incl_robot_node == false;
    @test scene_loader.curr_time_idx == start_time_idx + 1;
    @test controller.sim_param.dtc == dtc;
    @test controller.sim_param.dto == scene_loader.dto;
    @test controller.sim_param.prediction_steps == prediction_steps;
    @test controller.sim_param.num_samples == 0;
    @test controller.sim_param.cost_param.Cep == Cep;
    @test controller.sim_param.cost_param.Cu == Cu;
    @test controller.sim_param.cost_param.β_pos == β_pos;
    @test controller.sim_param.cost_param.α_col == α_col;
    @test controller.sim_param.cost_param.β_col == β_col;
    @test controller.sim_param.cost_param.λ_col == λ_col;
    @test controller.sim_param.cost_param.σ_risk == σ_risk;
    @test controller.cnt_param.model_dir == model_dir;
    @test controller.cnt_param.env_config == env_config;
    @test controller.cnt_param.policy_config == policy_config;
    @test controller.cnt_param.policy_name == policy_name;
    @test controller.cnt_param.goal_pos == ego_pos_goal_vec;
    @test controller.cnt_param.tcalc == tcalc;
    @test controller.cnt_param.dtr == dtr;
    @test measurement_schedule[1] == Time(controller.sim_param.dto);
    @test measurement_schedule[end] == Time(sim_horizon);
    @test minimum(target_trajectory)[1] == Time(0.0);
    @test minimum(target_trajectory)[2] == ego_pos_init_vec;
    @test maximum(target_trajectory)[1] == Time(14.8);
    @test maximum(target_trajectory)[2] == ego_pos_goal_vec;
    @test get_position(w_init.e_state) == ego_pos_init_vec;
    @test get_velocity(w_init.e_state) ==
        (ego_pos_goal_vec - ego_pos_init_vec)./
        norm(ego_pos_goal_vec - ego_pos_init_vec)*target_speed;
end

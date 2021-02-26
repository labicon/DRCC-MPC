#///////////////////////////////////////
#// File Name: rs_sac_controller_test.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/02/26
#// Description: Test code for src/rs_sac_controller.jl
#///////////////////////////////////////

using DataStructures
using LinearAlgebra
using Random
using RobotOS

@testset "RSSAC Controller Test" begin
# Cost parameters
Cep = Matrix(1.0I, 2, 2);
Cu = Matrix(1.0I, 2, 2);
β_pos = 0.6;
β_col = 0.4;
α_col = 100.0;
λ_col = 1.0;
σ_risk = 1.0;
cost_param = CostParameter(Cep, Cu, β_pos, α_col, β_col, λ_col, σ_risk);
# Simulation parameters
conf_file_name = "config.json";
test_data_name = "eth_test.pkl";
test_scene_id = 0;
start_time_idx = 50;
device = py"torch".device("cpu");
incl_robot_name = false;
scene_param = TrajectronSceneParameter(conf_file_name, test_data_name,
                                       test_scene_id, start_time_idx,
                                       incl_robot_name);
scene_loader = TrajectronSceneLoader(scene_param, verbose=false);
num_samples = 50;
prediction_steps = 12;
use_robot_future = false;
deterministic = false;
rng_seed_py = 1;
predictor_param = TrajectronPredictorParameter(prediction_steps, num_samples,
                                               use_robot_future, deterministic,
                                               rng_seed_py);
traj_predictor = TrajectronPredictor(predictor_param,
                                     scene_loader.model_dir,
                                     scene_loader.param.conf_file_name,
                                     device, verbose=false);
initialize_scene_graph!(traj_predictor, scene_loader);
ado_states = fetch_ado_positions!(scene_loader, return_full_state=true);
ado_positions = reduce_to_positions(ado_states);

dtc = 0.01;
sim_param = SimulationParameter(scene_loader, traj_predictor, dtc, cost_param);
# Initial conditions
ep = [-1., 3.5];
ev = [1., 1.];
t_init = Time(1.0);
e_init = RobotState([ep; ev], t_init);
ap_dict_init = convert_nodes_to_str(ado_positions);
t_last_m = Time(0.8);
w_init = WorldState(e_init, ap_dict_init, t_last_m);
# Target Trajectory
u_nominal_base = [0.0, 0.0];
u_array_base = [u_nominal_base for ii = 1:480];
wp1 = WayPoint2D([0.0, 0.0], Time(1.0));
wp2 = WayPoint2D([0.0, 0.0], Time(5, 8e8));
target_trajectory = Trajectory2D([wp1, wp2]);
# ControlParameter
eamax = 5.0;
tcalc = 0.1;
dtexec = [0.05, 0.01, 0.02];
dtr = 0.1;
nominal_search_depth = 2;
constraint_time = nothing;
u_nominal_cand = append!([[0.0, 0.0]],
                         [round.([a*cos(deg2rad(θ)), a*sin(deg2rad(θ))], digits=5)
                          for a = [2., 4.] for θ = 0.:45.:(360. - 45.)])             # nominal control candidate value [ax, ay] [m/s^2]
cnt_param = ControlParameter(eamax, tcalc, dtexec, dtr, u_nominal_base,
                             u_nominal_cand, nominal_search_depth,
                             constraint_time=constraint_time);
# Prediction Dict
prediction_dict = sample_future_ado_positions!(traj_predictor,
                                               ado_states);
num_controls = length(cnt_param.u_nominal_cand)^nominal_search_depth;
for key in keys(prediction_dict)
    prediction_dict[key] = repeat(prediction_dict[key], outer=(num_controls, 1, 1));
end
@test prediction_dict["PEDESTRIAN/25"][1:50, :, :] == prediction_dict["PEDESTRIAN/25"][51:100, :, :]

# # Helper functions test
# convert_to_schedule test
u_schedule = convert_to_schedule(w_init.t, u_array_base, sim_param);

@test length(u_schedule) == 480
@test all(collect(values(u_schedule)) .== [[0.0, 0.0]])
@test haskey(u_schedule, Time(1.0))
@test haskey(u_schedule, Time(1.01))
@test haskey(u_schedule, Time(5.79))

# get_nominal_u_arrays test: get nominal u_arrays from u_schedule and nominal control candidates
#u_nominal_mod_init_time = w_init.t + Duration(cnt_param.tcalc);
#u_nominal_mod_final_time = u_nominal_mod_init_time +
#                           Duration(sim_param.dto) -
#                           Duration(sim_param.dtc);
schedule_before = u_schedule;
u_arrays = get_nominal_u_arrays(u_schedule, sim_param, cnt_param);
@test all(u_arrays[1][1:end] .== [[0.0, 0.0]])
@test all([all(u_arrays[ii][1:10] .== [[0.0, 0.0]]) &&
           all(u_arrays[ii][11:50] .== [u_nominal_cand[div(ii - 1, length(cnt_param.u_nominal_cand)) + 1]]) &&
           all(u_arrays[ii][51:90] .== [u_nominal_cand[mod(ii - 1, length(cnt_param.u_nominal_cand)) + 1]]) &&
           all(u_arrays[ii][91:end] .== [[0.0, 0.0]])
           for ii = 2:length(cnt_param.u_nominal_cand)])
@test u_schedule == schedule_before

# get_robot_present_and_future_test
begin
    e_state = RobotState([-5.0, 0.0, 0.0, 1.0]);
    u_nominal = [[0.0, 0.0] for ii = 1:Int64(round(prediction_steps*sim_param.dto/dtc))];
    u_s = convert_to_schedule(e_state.t, u_nominal, sim_param);
    rpf = get_robot_present_and_future(e_state, u_s, sim_param, cnt_param);
    @test size(rpf) == (17^nominal_search_depth, 13, 6)
    @test all(isapprox.(cumsum(rpf[:, 1:end-1, 5:6], dims=2).*sim_param.dto .+ rpf[:, 1:1, 3:4], rpf[:, 2:end, 3:4], atol=1e-6))

end

# Get simulation result and best nominal u_array
sim_result, u_nominal_array = simulate(w_init, u_arrays, target_trajectory,
                                       prediction_dict, sim_param, cnt_param.constraint_time);

# get_control_coeffs_test
coeff_matrix, coeff_matrix_constraint = get_control_coeffs(sim_result, sim_param, cnt_param);
#@test size(coeff_matrix) == (2, 20);
@test size(coeff_matrix) == (2, 480);
@test isnothing(coeff_matrix_constraint)

test_time_id = 11;
den_test = sum(exp.(σ_risk.*sim_result.sampled_total_costs));
num_test = sum(exp.(σ_risk.*sim_result.sampled_total_costs).*
               [sim_result.e_costate_array[:, test_time_id, jj] for jj = 1:sim_param.num_samples]);

@test transition_control_coeff(sim_result.e_state_array[test_time_id])'*
    (num_test./den_test) ≈ coeff_matrix[:, test_time_id];

# get_control_schedule test
control_schedule_array = get_control_schedule(sim_result, u_nominal_array, coeff_matrix,
                                              sim_param, cnt_param);
#@test length(control_schedule_array) == Int64((tcalc+dtr)/dtc);
@test length(control_schedule_array) == Int64(round(sim_param.dto*sim_param.prediction_steps/dtc, digits=5));
@test maximum(map(s -> norm(vec(s.u)), control_schedule_array)) < eamax ||
      maximum(map(s -> norm(vec(s.u)), control_schedule_array)) ≈ eamax;
@test !any(isnan.(map(s -> s.cost, control_schedule_array)));

# determine_control_time test
control_chosen = determine_control_time(sim_result,
                                        control_schedule_array, sim_param,
                                        cnt_param);
@test t_init + Duration(cnt_param.tcalc) + Duration(maximum(cnt_param.dtexec)) <= control_chosen.t;
#@test control_chosen.t <= t_init + Duration(cnt_param.tcalc) + Duration(cnt_param.dtr)
@test control_chosen.t <= t_init + Duration(sim_param.dto*sim_param.prediction_steps);
@test norm(control_chosen.u) < eamax || norm(control_chosen.u) ≈ eamax;

# sac_control_update_test
tcalc_actual, best_schedule, sim_result =
    sac_control_update(w_init, u_schedule, target_trajectory, prediction_dict,
                       sim_param, cnt_param);
perturbation_times = [k for (k,v) in best_schedule if v==control_chosen.u];
if length(perturbation_times) > 1
    @test best_schedule[control_chosen.t - Duration(dtc)] == control_chosen.u
    @test best_schedule[control_chosen.t] != control_chosen.u
    @test in(to_sec(perturbation_times[end] - perturbation_times[1]) + sim_param.dtc, cnt_param.dtexec)
end

# # Main functions test
u_schedule = convert_to_schedule(w_init.t, u_array_base, sim_param);

controller = RSSACController(traj_predictor, u_schedule, sim_param, cnt_param);
scene_loader = TrajectronSceneLoader(scene_param, verbose=false);
traj_predictor = TrajectronPredictor(predictor_param,
                                     scene_loader.model_dir,
                                     scene_loader.param.conf_file_name,
                                     device, verbose=false);
initialize_scene_graph!(traj_predictor, scene_loader);
ado_states = fetch_ado_positions!(scene_loader, return_full_state=true);
ado_positions = reduce_to_positions(ado_states);

schedule_prediction!(controller, ado_states);
wait(controller.prediction_task);
@test istaskdone(controller.prediction_task);
@test controller.prediction_dict_tmp["PEDESTRIAN/25"][1:50, :, :] ==
      controller.prediction_dict_tmp["PEDESTRIAN/25"][51:100, :, :]

schedule_control_update!(controller, w_init, target_trajectory);
@test !isnothing(controller.prediction_dict);
@test isnothing(controller.prediction_task);
wait(controller.control_update_task);
@test istaskdone(controller.control_update_task);

ado_positions_str = convert_nodes_to_str(ado_positions);
latest_ado_pos_dict = deepcopy(ado_positions_str);
ado_1 = collect(keys(latest_ado_pos_dict))[1];
ado_2 = collect(keys(latest_ado_pos_dict))[2];
prediction_array_ado_2 = copy(controller.prediction_dict[ado_2]);
latest_ado_pos_dict[ado_2] += [0.5, 0.5];
pop!(latest_ado_pos_dict, ado_1);
latest_ado_pos_dict["PEDESTRIAN/999"] = [1.0, 1.0];
adjust_old_prediction!(controller, ado_positions_str, latest_ado_pos_dict);
@test all(controller.prediction_dict[ado_2][:, :, 1] .- prediction_array_ado_2[:, :, 1] .≈ 0.5);
@test all(controller.prediction_dict[ado_2][:, :, 2] .- prediction_array_ado_2[:, :, 2] .≈ 0.5);
@test !in(controller.prediction_dict, ado_1);
@test size(controller.prediction_dict["PEDESTRIAN/999"]) == (num_samples*num_controls, sim_param.prediction_steps, 2);
@test all(controller.prediction_dict["PEDESTRIAN/999"][:, :, 1] .== 1.0)
@test all(controller.prediction_dict["PEDESTRIAN/999"][:, :, 2] .== 1.0)


u_1 = control!(controller, Time(1.0));
@test u_1 == [0.0, 0.0];
@test !isnothing(controller.sim_result);
@test !isnothing(controller.tcalc_actual);
#@test !isnothing(controller.u_init_time);
#@test !isnothing(controller.u_last_time);
#@test !isnothing(controller.u_value);
@test isnothing(controller.control_update_task);

# # constraint_time tests
constraint_time = 0.1;
cnt_param = ControlParameter(eamax, tcalc, dtexec, dtr, u_nominal_base,
                             u_nominal_cand, nominal_search_depth,
                             constraint_time=constraint_time);
# Get simulation result and best nominal u_array
sim_result, u_nominal_array = simulate(w_init, u_arrays, target_trajectory,
                                       prediction_dict, sim_param, cnt_param.constraint_time);
# get_control_coeffs_test
coeff_matrix, coeff_matrix_constraint = get_control_coeffs(sim_result, sim_param, cnt_param);
#@test size(coeff_matrix) == (2, 20);
@test size(coeff_matrix) == (2, 480);
@test !isnothing(coeff_matrix_constraint)
@test size(coeff_matrix_constraint) == (2, 10);

u_array_constraint, cost_array_constraint = solve_multi_qcqp(u_nominal_array,
                                                             coeff_matrix,
                                                             coeff_matrix_constraint,
                                                             sim_param, cnt_param);
@test all([norm(u) <= cnt_param.eamax for u in u_array_constraint])
control_schedule_array_constraint =
    get_control_schedule(sim_result, u_nominal_array, coeff_matrix,
                         sim_param, cnt_param, coeff_matrix_constraint);
@test all([s.u == u for (s, u) in zip(control_schedule_array_constraint[1:10], u_array_constraint)])
@test all([s.cost == cost for (s, cost) in zip(control_schedule_array_constraint[1:10], cost_array_constraint)])
@test all([s.t == s_constraint.t for (s, s_constraint) in zip(control_schedule_array[1:10], control_schedule_array_constraint[1:10])])
@test all([s.u == s_constraint.u for (s, s_constraint) in zip(control_schedule_array[11:20], control_schedule_array_constraint[11:20])])
@test all([s.cost == s_constraint.cost for (s, s_constraint) in zip(control_schedule_array[11:20], control_schedule_array_constraint[11:20])])
@test all([s.t == s_constraint.t for (s, s_constraint) in zip(control_schedule_array[11:20], control_schedule_array_constraint[11:20])])
# sac_control_update_test
tcalc_actual, best_schedule, sim_result =
    sac_control_update(w_init, u_schedule, target_trajectory, prediction_dict,
                       sim_param, cnt_param);
end

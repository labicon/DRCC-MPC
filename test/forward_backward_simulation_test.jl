#///////////////////////////////////////
#// File Name: forwawrd_backward_simulation_test.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/02/26
#// Description: Test code for src/forward_backward_simulation.jl
#///////////////////////////////////////

using CUDA
using Distributions
using LinearAlgebra
using Random
using RobotOS

@testset "Forward Backward Simulation Test" begin
prediction_device = "cpu";                                                          # "cpu" or "cuda"
prediction_steps = 12;                                                              # number of steps to look ahead in the future
dto = 0.4;                                                                          # observation update time interval [s]
deterministic = false;
Cep = Matrix(1.0I, 2, 2);                                                           # quadratic position cost matrix
Cu = Matrix(1.0I, 2, 2);                                                            # quadratic control cost matrix
β_pos = 0.6;                                                                        # relative weight between instant and terminal pos cost
β_col = 0.4;                                                                        # relative weight between instant and terminal col cost
α_col = 50.0;                                                                       # scale parameter for exponential collision cost
λ_col = 1.0;                                                                        # bandwidth parameter for exponential collision cost
σ_risk = 0.01;                                                                       # risk-sensitiveness parameteru_norm_max = 2.0;                                                                   # maximum control norm [m/s^2]
dtc = 0.02;                                                                         # Euler integration time interval [s]
dtr = 0.1;                                                                          # replanning time interval [s]
dtexec = [0.02, 0.04, 0.06];                                                        # control insertion duration candidates [s]
tcalc = 0.1;                                                                        # pre-allocated control computation time [s] (< dtr)
u_norm_max = 5.0;                                                                   # maximum control norm [m/s^2]
u_nominal_base = [0.0, 0.0];                                                        # baseline nominal control [m/s^2]
u_nominal_cand = append!([u_nominal_base],
                         [round.([a*cos(deg2rad(θ)), a*sin(deg2rad(θ))], digits=5)
                          for a = [2., 4.] for θ = 0.:45.:(360. - 45.)])            # nominal control candidate value [ax, ay] [m/s^2]
nominal_search_depth = 1;
# Ego initial state
ego_pos_init_vec = [-5., 0.];                                                       # initial ego position [x, y] [m]
ego_pos_goal_vec = [5., 0.];                                                        # goal ego position [x, y] [m]
# Other parameters
target_speed = 1.0;                                                              # target trajectory horizon [s]
sim_horizon = 10.0;                                                                 # simulation time horizon [s]

num_samples = 50;
num_ado_agents = 3;

# Ado positions
apos_shared = [0.0, -5.0]
ado_pos_init_dict = Dict{String, Vector{Float64}}();
for ii = 1:num_ado_agents
    ado_pos_init_dict["PEDESETRIAN/$(ii)"] = apos_shared
end
ado_vel_dict = Dict("Any" => MvNormal([0.0, 0.0], Diagonal([0.8, 0.8])));
prediction_rng_seed = 1;

rng = MersenneTwister(prediction_rng_seed);
scene_param = SyntheticSceneParameter(rng);
predictor_param = GaussianPredictorParameter(prediction_steps,
                                             num_samples, deterministic, rng);
cost_param = CostParameter(Cep, Cu, β_pos, α_col, β_col, λ_col, σ_risk);
cnt_param = ControlParameter(u_norm_max, tcalc, dtexec, dtr,
                             [0.0, 0.0], u_nominal_cand, nominal_search_depth);

scene_loader, controller, w_init, measurement_schedule =
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

# Nominal Controls
u_array_1 = [[0.0, 0.0] for ii = 1:240];
u_array_2 = [[0.1, 0.05] for ii = 1:240];

u_arrays = cat(repeat([u_array_1], inner=12),
             repeat([u_array_2], inner=12), dims=1);

# Helper functions test
# # process_u_arrays test
u_array = process_u_arrays(u_arrays);
u_array_gpu = cu(u_array)
@test u_array[1, 1, :] == u_arrays[1][1]
@test u_array[13, 200, :] == u_arrays[13][200]

# # simulate_forward test
e_state_array_1 = simulate_forward(w_init.e_state, u_array_1, controller.sim_param)
@test length(e_state_array_1) == length(u_array_1) + 1;
@test e_state_array_1[1] == w_init.e_state;
@test to_sec(e_state_array_1[end].t) ≈
        to_sec(w_init.e_state.t) + dtc*length(u_array_1);
# # simulate_forward test (CUDA version)
e_state_array_2 = simulate_forward(w_init.e_state, u_array_2, controller.sim_param);
e_state_array = cat(repeat(reshape(e_state_array_1, (1, length(e_state_array_1))), inner=(12, 1)),
                    repeat(reshape(e_state_array_2, (1, length(e_state_array_2))), inner=(12, 1)),
                    dims=1);
ex_array_gpu = simulate_forward(w_init.e_state, u_array_gpu, controller.sim_param);
ex_array_cpu = collect(ex_array_gpu);
ex_array = Array{Float64, 3}(undef, size(e_state_array, 1), size(e_state_array, 2), 4);
for ii = 1:size(ex_array, 1)
    for jj = 1:size(ex_array, 2)
        for kk = 1:size(ex_array, 3)
            ex_array[ii, jj, kk] = e_state_array[ii, jj].x[kk]
        end
    end
end
@test all(isapprox.(ex_array_cpu, ex_array, atol=5e-5))

# # get_measurement_schedule test
m_schedule_test = get_measurement_schedule(w_init, 4.4, controller.sim_param);
@test length(m_schedule_test) == 11;
@test to_sec(m_schedule_test[1]) ≈ 0.0 + controller.sim_param.dto;
@test to_sec(m_schedule_test[2]) ≈ 0.0 + 2*controller.sim_param.dto;
@test to_sec(m_schedule_test[end]) ≈ 0.0 + 4.4;
measurement_schedule = get_measurement_schedule(w_init, controller.sim_param);
@test length(measurement_schedule) == 12;
@test to_sec(measurement_schedule[1]) ≈ 0.0 + controller.sim_param.dto;
@test to_sec(measurement_schedule[2]) ≈ 0.0 + 2*controller.sim_param.dto;\
@test to_sec(measurement_schedule[end]) ≈ 0.0 + controller.sim_param.prediction_steps*controller.sim_param.dto;

# # process_ap_dict test
prediction_dict = sample_future_ado_positions!(controller.predictor, ado_pos_init_dict)
for key in keys(prediction_dict)
    prediction_dict[key] = repeat(prediction_dict[key], outer=(size(u_array_gpu, 1), 1, 1))
end
ap_array, time_idx_ap_array, control_idx_ex_array, ado_ids_array =
    process_ap_dict(ex_array_gpu, w_init, measurement_schedule, prediction_dict, controller.sim_param);
ped_1_key = collect(keys(w_init.ap_dict))[1];
ped_1_idx = findfirst(isequal(ped_1_key), ado_ids_array);
@test size(ap_array) == (controller.sim_param.num_samples*size(u_array_gpu, 1),
                         controller.sim_param.prediction_steps + 1, length(prediction_dict), 2);
@test ap_array[1, 1, ped_1_idx, :] == vec(w_init.ap_dict[ped_1_key])
init_timesteps = Int64(0.4/dtc);
total_timesteps = length(e_state_array_1);
expansion_factor = Int64(controller.sim_param.dto/controller.sim_param.dtc);
@test ap_array[1, time_idx_ap_array[init_timesteps], ped_1_idx, :] == vec(w_init.ap_dict[ped_1_key])
@test all([ap_array[1, time_idx_ap_array[jj], ped_1_idx, :] ==
            prediction_dict[ped_1_key][1, 1, :]
            for jj = init_timesteps + 1:init_timesteps + expansion_factor]);
@test ap_array[13, time_idx_ap_array[init_timesteps + expansion_factor + 1], ped_1_idx, :] ==
        prediction_dict[ped_1_key][13, 2, :]
@test ap_array[13, time_idx_ap_array[init_timesteps + (controller.sim_param.prediction_steps - 1)*expansion_factor], ped_1_idx, :] ==
        prediction_dict[ped_1_key][13, end-1, :]
@test ap_array[13, time_idx_ap_array[init_timesteps + (controller.sim_param.prediction_steps - 1)*expansion_factor + 1], ped_1_idx, :] ==
        prediction_dict[ped_1_key][13, end, :]
@test ap_array[13, time_idx_ap_array[end], ped_1_idx, :] == prediction_dict[ped_1_key][13, end, :]
@test all(control_idx_ex_array[1:50] .== 1)
@test all(control_idx_ex_array[51:100] .== 2)
@test length(control_idx_ex_array) == 24*50

# # get_target_pos_array test
prediction_horizon = dto*prediction_steps;
target_trajectory = get_nominal_trajectory(w_init.e_state,
                                           ego_pos_goal_vec,
                                           target_speed,
                                           sim_horizon,
                                           prediction_horizon);
target_pos_array = get_target_pos_array(ex_array_gpu, w_init, target_trajectory, controller.sim_param)
@test size(target_pos_array) == (size(ex_array, 2), 2)

# # compute_costs test
ap_array_gpu = cu(ap_array)
target_pos_array_gpu = cu(target_pos_array);
time_idx_ap_array_gpu = CuArray{Int32}(time_idx_ap_array);
control_idx_ex_array_gpu = CuArray{Int32}(control_idx_ex_array);
cost_result = compute_costs(ex_array_gpu, u_array_gpu, ap_array_gpu,
                             time_idx_ap_array_gpu, control_idx_ex_array_gpu,
                             target_pos_array_gpu, controller.sim_param.cost_param);
@test size(cost_result.inst_cnt_cost_array_gpu) == (24, 240);
@test size(cost_result.inst_pos_cost_array_gpu) == (24, 240);
@test size(cost_result.inst_col_cost_array_gpu) == (50*24, 240, 3)
@test size(cost_result.term_pos_cost_array_gpu) == (24, 1);
@test size(cost_result.term_col_cost_array_gpu) == (50*24, 1, 3)

# # integrate_costs test
total_cost_per_control_sample, risk_per_control = integrate_costs(cost_result, controller.sim_param);
total_cost_per_control_sample_test = similar(total_cost_per_control_sample)
for ii = 1:size(total_cost_per_control_sample_test, 1)
    for jj = 1:size(total_cost_per_control_sample_test, 2)
        total_cost_per_control_sample_test[ii, jj] =
             sum(collect(cost_result.inst_cnt_cost_array_gpu), dims=2)[ii]*controller.sim_param.dtc +
             sum(collect(cost_result.inst_pos_cost_array_gpu), dims=2)[ii]*controller.sim_param.dtc +
             collect(cost_result.term_pos_cost_array_gpu)[ii] +
             sum(collect(cost_result.inst_col_cost_array_gpu), dims=(2, 3))[50*(ii - 1) + jj]*controller.sim_param.dtc +
             sum(collect(cost_result.term_col_cost_array_gpu), dims=(2, 3))[50*(ii - 1) + jj];
    end
end
@test all(isapprox.(total_cost_per_control_sample, total_cost_per_control_sample_test, atol=5e-5))
risk_per_control_test = similar(risk_per_control)
for ii = 1:length(risk_per_control)
    risk_val = 1/controller.sim_param.cost_param.σ_risk*
               log(1/length(total_cost_per_control_sample[ii, :])*
               sum(exp.(controller.sim_param.cost_param.σ_risk.*total_cost_per_control_sample[ii, :])));
    risk_per_control_test[ii] = risk_val
end
@test all(isapprox.(risk_per_control, risk_per_control_test, atol=5e-5));

# # integrate_costs_test with constraint_time;
constraint_time = 0.4;
total_cost_per_control_sample_constraint, risk_per_control_constraint =
integrate_costs(cost_result, controller.sim_param, constraint_time);
@test size(total_cost_per_control_sample_constraint) == size(total_cost_per_control_sample);
@test size(risk_per_control_constraint) == size(risk_per_control);

# choose_best_nominal_control test
sampled_total_costs, minimum_risk, best_control_idx, best_u_array =
    choose_best_nominal_control(total_cost_per_control_sample, risk_per_control, u_array_gpu);
@test minimum_risk == minimum(risk_per_control)
@test risk_per_control[best_control_idx] == minimum(risk_per_control);
@test total_cost_per_control_sample[best_control_idx, :] == sampled_total_costs
@test best_u_array == [Float64.(collect(u_array_gpu[best_control_idx, ii, :])) for ii = 1:size(u_array_gpu, 2)]

# # compute_cost_gradients test
best_ex_array_gpu = ex_array_gpu[best_control_idx, :, :];
best_ex_array = Float64.(collect(best_ex_array_gpu));
@test all(isapprox.(best_ex_array, permutedims(hcat([s.x for s in simulate_forward(w_init.e_state, best_u_array, controller.sim_param)]...),
                                               [2, 1]), atol=5e-5))
best_u_array_gpu = u_array_gpu[best_control_idx, :, :];
best_ap_array_sample_idx = controller.sim_param.num_samples*(best_control_idx-1)+1:controller.sim_param.num_samples*best_control_idx
best_ap_array_gpu = ap_array_gpu[best_ap_array_sample_idx, :, :, :];
cost_grad_result = compute_cost_gradients(best_ex_array_gpu, best_u_array_gpu,
                                          best_ap_array_gpu, time_idx_ap_array_gpu,
                                          target_pos_array_gpu,
                                          controller.sim_param.cost_param);
@test size(cost_grad_result.inst_pos_cost_grad_array_gpu) == (240, 4);
@test size(cost_grad_result.inst_col_cost_grad_array_gpu) == (50, 240, 3, 4);
@test size(cost_grad_result.term_pos_cost_grad_array_gpu) == (1, 4);
@test size(cost_grad_result.term_col_cost_grad_array_gpu) == (50, 1, 3, 4)

# # sum_cost_gradients test
cost_grad_array = sum_cost_gradients(cost_grad_result);
cost_grad_array_test = similar(cost_grad_array);
for jj = best_ap_array_sample_idx # iterate over samples
    for kk = 1:size(cost_grad_array, 2) - 1 # iterate over timesteps
        mm = time_idx_ap_array[kk];
        cost_grad = instant_position_cost_gradient(e_state_array[best_control_idx, kk], target_trajectory, controller.sim_param.cost_param) +
                                                   sum([instant_collision_cost_gradient(e_state_array[best_control_idx, kk], ap_array[jj, mm, ii, :], controller.sim_param.cost_param)
                                                   for ii = 1:size(ap_array, 3)]);
        cost_grad_array_test[:, kk, jj] = cost_grad
    end
    cost_grad = terminal_position_cost_gradient(e_state_array[best_control_idx, end], target_trajectory, controller.sim_param.cost_param) +
                sum([terminal_collision_cost_gradient(e_state_array[best_control_idx, end], ap_array[jj, end, ii, :], controller.sim_param.cost_param)
                     for ii = 1:size(ap_array, 3)]);
    cost_grad_array_test[:, end, jj] = cost_grad
end
@test all(isapprox.(cost_grad_array, cost_grad_array_test, atol=5e-5))

# # simulate_backward test
e_costate_array = simulate_backward(best_ex_array, best_u_array, cost_grad_array, controller.sim_param);
test_time_id = 151;
test_sample_id = 27;
@test e_costate_array[:, end, test_sample_id] ==
        cost_grad_array[:, end, test_sample_id]
J = transition_jacobian(e_state_array[best_control_idx, test_time_id],
                        u_arrays[best_control_idx][test_time_id - 1]);
costate_vel = -cost_grad_array[:, test_time_id - 1,  test_sample_id] -
              transpose(J)*e_costate_array[:, test_time_id, test_sample_id];
@test e_costate_array[:, test_time_id - 1, test_sample_id] ≈
        e_costate_array[:, test_time_id, test_sample_id] +
        costate_vel*(-dtc)

# Main function test (make sure there is no error)
sim_result, u_nominal_array =
    simulate(w_init, u_arrays, target_trajectory, prediction_dict,
             controller.sim_param);
@test isnothing(sim_result.e_costate_array_constraint)

# Main function test with constraint_time
constraint_time = 0.4;
sim_result, u_nominal_array =
    simulate(w_init, u_arrays, target_trajectory, prediction_dict,
             controller.sim_param, constraint_time);
@test !isnothing(sim_result.e_costate_array_constraint)
@test size(sim_result.e_costate_array_constraint) ==  (4, Int64(dto/dtc) + 1, num_samples)
@test all(sim_result.e_costate_array_constraint[:, end, :] .== 0.0)

#=
# forward (re-) simulation test
risk_test = evaluate_risk(w_init, u_array, target_trajectory, ap_array_gpu, sim_param);
@test risk_test ≈ risk_value
=#
end

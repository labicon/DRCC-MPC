#///////////////////////////////////////
#// File Name: cost_gpu_test.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/01/15
#// Description: Test code for src/cost_gpu.jl
#///////////////////////////////////////

using CUDA
using Distributions
using LinearAlgebra
using Random
using RobotOS

@testset "Cost CUDA Test" begin
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
σ_risk = 0.0;                                                                       # risk-sensitiveness parameteru_norm_max = 2.0;
dtc = 0.02;                                                                         # Euler integration time interval [s]
dtr = 0.1;                                                                          # replanning time interval [s]
dtexec = [0.02, 0.06];                                                              # control insertion duration [s]
tcalc = 0.1;                                                                        # pre-allocated control computation time [s] (< dtr)
u_norm_max = 5.0;                                                                   # maximum control norm [m/s^2]
u_nominal_cand = append!([[0.0, 0.0]],
                         [round.([a*cos(deg2rad(θ)), a*sin(deg2rad(θ))], digits=5)
                          for a = [2., 4.] for θ = 0.:45.:(360. - 45.)])            # nominal control candidate value [ax, ay] [m/s^2]
nominal_search_depth = 1;
# Ego initial state
ego_pos_init_vec = [-5., 0.];                                                       # initial ego position [x, y] [m]
ego_pos_goal_vec = [5., 0.];                                                        # initial ego velocity [vx, vy] [m/s]
# Other parameters
target_speed = 1.0;
sim_horizon = 10.0;                                                                 # simulation time horizon [s]


num_samples = 50;
num_ado_agents = 3;


apos_shared = [0.0, -5.0]
ado_pos_init_dict = Dict{String, Vector{Float64}}();
for ii = 1:num_ado_agents
    ado_pos_init_dict["PEDESTRIAN/$(ii)"] = apos_shared
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
u_array = cat(repeat(reshape(u_array_1, (1, length(u_array_1))), inner=(12, 1)),
              repeat(reshape(u_array_2, (1, length(u_array_2))), inner=(12, 1)),
              dims=1);

u_array_gpu_1 = reshape(transpose(hcat(u_array_1...)), 1, 240, 2);
u_array_gpu_1 = cu(repeat(u_array_gpu_1, inner=(12, 1, 1)));
u_array_gpu_2 = reshape(transpose(hcat(u_array_2...)), 1, 240, 2);
u_array_gpu_2 = cu(repeat(u_array_gpu_2, inner=(12, 1, 1)));
u_array_gpu = cat(u_array_gpu_1, u_array_gpu_2, dims=1);

e_state_array_1 = simulate_forward(w_init.e_state, u_array_1, controller.sim_param)
e_state_array_2 = simulate_forward(w_init.e_state, u_array_2, controller.sim_param);
e_state_array = cat(repeat(reshape(e_state_array_1, (1, length(e_state_array_1))), inner=(12, 1)),
                    repeat(reshape(e_state_array_2, (1, length(e_state_array_2))), inner=(12, 1)),
                    dims=1);

ex_array_cpu = Array{Float64, 3}(undef, size(e_state_array, 1), size(e_state_array, 2), 4);
for ii = 1:size(ex_array_cpu, 1)
    for jj = 1:size(ex_array_cpu, 2)
        for kk = 1:size(ex_array_cpu, 3)
            ex_array_cpu[ii, jj, kk] = e_state_array[ii, jj].x[kk]
        end
    end
end
ex_array_gpu = cu(ex_array_cpu);
prediction_horizon = dto*prediction_steps;
target_trajectory = get_nominal_trajectory(w_init.e_state,
                                           ego_pos_goal_vec,
                                           target_speed,
                                           sim_horizon,
                                           prediction_horizon);
target_pos_array = get_target_pos_array(ex_array_gpu, w_init, target_trajectory,
                                        controller.sim_param);
target_pos_array_gpu = cu(target_pos_array);


# Instant Position Cost
inst_pos_cost_array_gpu = instant_position_cost(ex_array_gpu, target_pos_array_gpu,
                                                controller.sim_param.cost_param);
inst_pos_cost_array_cpu = collect(inst_pos_cost_array_gpu);
inst_pos_cost_array =
        map(s -> instant_position_cost(s, target_trajectory, controller.sim_param.cost_param), e_state_array[:, 1:end-1]);
@test all(isapprox.(inst_pos_cost_array_cpu, inst_pos_cost_array, atol=5e-5));

# Instant Control Cost
inst_cnt_cost_array_gpu = instant_control_cost(u_array_gpu, controller.sim_param.cost_param);
inst_cnt_cost_array_cpu = collect(inst_cnt_cost_array_gpu);
inst_cnt_cost_array =
        map(u -> instant_control_cost(u,controller.sim_param.cost_param), u_array);
@test all(isapprox.(inst_cnt_cost_array_cpu, inst_cnt_cost_array, atol=5e-5));

# Instant Collision Cost
prediction_dict = sample_future_ado_positions!(controller.predictor, ado_pos_init_dict);
for key in keys(prediction_dict)
    prediction_dict[key] = repeat(prediction_dict[key], outer=(size(u_array_gpu, 1), 1, 1))
end

ap_array, time_idx_ap_array, control_idx_ex_array, ado_ids_array =
    process_ap_dict(ex_array_gpu, w_init, measurement_schedule, prediction_dict, controller.sim_param);

ap_array_gpu = cu(ap_array);
time_idx_ap_array_gpu = CuArray{Int32, 1}(time_idx_ap_array);
control_idx_ex_array_gpu = CuArray{Int32, 1}(control_idx_ex_array);

inst_col_cost_array_gpu = instant_collision_cost(ex_array_gpu, ap_array_gpu,
                                                 time_idx_ap_array_gpu, control_idx_ex_array_gpu,
                                                 controller.sim_param.cost_param);
inst_col_cost_array_cpu = collect(inst_col_cost_array_gpu);
inst_col_cost_array = similar(inst_col_cost_array_cpu);
for ii = 1:size(inst_col_cost_array, 1)
    for jj = 1:size(inst_col_cost_array, 2)
        for kk = 1:size(inst_col_cost_array, 3)
            ll = control_idx_ex_array[ii];
            mm = time_idx_ap_array[jj];
            inst_col_cost_array[ii, jj, kk] =
                instant_collision_cost(e_state_array[ll, jj], ap_array[ii, mm, kk, :],
                                      controller.sim_param.cost_param);
        end
    end
end
@test all(isapprox.(inst_col_cost_array, inst_col_cost_array_cpu, atol=5e-5))
@test all(isapprox(inst_col_cost_array_cpu[1:50, :, :], inst_col_cost_array_cpu[51:100, :, :], atol=1e-10))
# Instanat Collision Cost with 0 pedestrians
ap_array_gpu_zeros = CuArray{Float32, 4}(undef, size(ap_array_gpu, 1), size(ap_array_gpu, 2), 0, 2);
inst_col_cost_array_gpu_zeros = instant_collision_cost(ex_array_gpu, ap_array_gpu_zeros,
                                                       time_idx_ap_array_gpu, control_idx_ex_array_gpu,
                                                       controller.sim_param.cost_param);
@test size(inst_col_cost_array_gpu_zeros) == (size(inst_col_cost_array_gpu, 1),
                                              size(inst_col_cost_array_gpu, 2),
                                              0);

# Terminal Position Cost
term_pos_cost_array_gpu = terminal_position_cost(ex_array_gpu, target_pos_array_gpu,
                                                 controller.sim_param.cost_param);
term_pos_cost_array_cpu = collect(term_pos_cost_array_gpu);
term_pos_cost_array =
        map(s -> terminal_position_cost(s, target_trajectory, controller.sim_param.cost_param), e_state_array[:, end:end]);
@test all(isapprox.(term_pos_cost_array_cpu, term_pos_cost_array, atol=5e-5))

# Terminal Collision Cost
term_col_cost_array_gpu = terminal_collision_cost(ex_array_gpu, ap_array_gpu,
                                                  time_idx_ap_array_gpu,
                                                  control_idx_ex_array_gpu,
                                                  controller.sim_param.cost_param);
term_col_cost_array_cpu = collect(term_col_cost_array_gpu);
term_col_cost_array = similar(term_col_cost_array_cpu);
for ii = 1:size(inst_col_cost_array, 1)
    for kk = 1:size(inst_col_cost_array, 3)
        ll = control_idx_ex_array[ii];
        mm = time_idx_ap_array[end];
        term_col_cost_array[ii, end, kk] =
            terminal_collision_cost(e_state_array[ll, end], ap_array[ii, mm, kk, :],
                                    controller.sim_param.cost_param);
    end
end
@test all(isapprox.(term_col_cost_array, term_col_cost_array_cpu, atol=5e-5))
@test all(isapprox(term_col_cost_array_cpu[1:50, :, :], term_col_cost_array_cpu[51:100, :, :], atol=1e-10))
# Terminal Collision Cost with 0 pedestrians
term_col_cost_array_gpu_zeros = terminal_collision_cost(ex_array_gpu, ap_array_gpu_zeros,
                                                        time_idx_ap_array_gpu,
                                                        control_idx_ex_array_gpu,
                                                        controller.sim_param.cost_param);
@test size(term_col_cost_array_gpu_zeros) == (size(term_col_cost_array_gpu, 1),
                                              size(term_col_cost_array_gpu, 2),
                                              0);

# Instant Position Cost Gradient
inst_pos_cost_grad_array_gpu = instant_position_cost_gradient(ex_array_gpu[13, :, :],
                                                              target_pos_array_gpu,
                                                              controller.sim_param.cost_param);
inst_pos_cost_grad_array_cpu = collect(inst_pos_cost_grad_array_gpu)

inst_pos_cost_grad_array =
        map(s -> instant_position_cost_gradient(s, target_trajectory, controller.sim_param.cost_param), e_state_array[13, 1:end-1]);
inst_pos_cost_grad_array = Array(transpose(hcat(inst_pos_cost_grad_array...)));
@test all(isapprox.(inst_pos_cost_grad_array_cpu, inst_pos_cost_grad_array, atol=5e-5))

# Instant Collision Cost Gradient
inst_col_cost_grad_array_gpu = instant_collision_cost_gradient(ex_array_gpu[1, :, :],
                                                               ap_array_gpu[1:50, :, :, :],
                                                               time_idx_ap_array_gpu,
                                                               controller.sim_param.cost_param);
inst_col_cost_grad_array_cpu = collect(inst_col_cost_grad_array_gpu);
inst_col_cost_grad_array = similar(inst_col_cost_grad_array_cpu);
for ii = 1:size(inst_col_cost_grad_array, 1)
    for jj = 1:size(inst_col_cost_grad_array, 2)
        for kk = 1:size(inst_col_cost_grad_array, 3)
            mm = time_idx_ap_array[jj];
            inst_col_cost_grad_array[ii, jj, kk, :] =
                instant_collision_cost_gradient(e_state_array[1, jj], ap_array[ii, mm, kk, :],
                                                controller.sim_param.cost_param);
        end
    end
end
@test all(isapprox.(inst_col_cost_grad_array_cpu, inst_col_cost_grad_array, atol=5e-5))
# Instant Collision Cost Gradient with 0 pedestrians
inst_col_cost_grad_array_gpu_zeros = instant_collision_cost_gradient(ex_array_gpu[1, :, :],
                                                                     ap_array_gpu_zeros[1:50, :, :, :],
                                                                     time_idx_ap_array_gpu,
                                                                     controller.sim_param.cost_param);
@test size(inst_col_cost_grad_array_gpu_zeros) == (size(inst_col_cost_grad_array_gpu, 1),
                                                   size(inst_col_cost_grad_array_gpu, 2),
                                                   0,
                                                   size(inst_col_cost_grad_array_gpu, 4));

# Terminal Position Cost Gradient
term_pos_cost_grad_array_gpu = terminal_position_cost_gradient(ex_array_gpu[13, :, :], target_pos_array_gpu,
                                                           controller.sim_param.cost_param);
term_pos_cost_grad_array_cpu = collect(term_pos_cost_grad_array_gpu);
term_pos_cost_grad_array = terminal_position_cost_gradient(e_state_array[13, end], target_trajectory,
                                                           controller.sim_param.cost_param);
term_pos_cost_grad_array = Array(transpose(term_pos_cost_grad_array));
@test all(isapprox(term_pos_cost_grad_array_cpu, term_pos_cost_grad_array, atol=5e-5));

# Terminal Collision Cost Gradient
term_col_cost_grad_array_gpu = terminal_collision_cost_gradient(ex_array_gpu[1, :, :],
                                                                ap_array_gpu[1:50, :, :, :],
                                                                time_idx_ap_array_gpu,
                                                                controller.sim_param.cost_param);
term_col_cost_grad_array_cpu = collect(term_col_cost_grad_array_gpu);
term_col_cost_grad_array = similar(term_col_cost_grad_array_cpu);
for ii = 1:size(inst_col_cost_grad_array, 1)
    for kk = 1:size(inst_col_cost_grad_array, 3)
        mm = time_idx_ap_array[end];
        term_col_cost_grad_array[ii, end, kk, :] =
        terminal_collision_cost_gradient(e_state_array[1, end], ap_array[ii, mm, kk, :],
                                         controller.sim_param.cost_param);
    end
end
@test all(isapprox.(term_col_cost_grad_array_cpu, term_col_cost_grad_array, atol=5e-5))
# Terminal Collision Cost Gradient with 0 pedestrians
term_col_cost_grad_array_gpu_zeros = terminal_collision_cost_gradient(ex_array_gpu[1, :, :],
                                                                      ap_array_gpu_zeros[1:50, :, :, :],
                                                                      time_idx_ap_array_gpu,
                                                                      controller.sim_param.cost_param);
@test size(term_col_cost_grad_array_gpu_zeros) == (size(term_col_cost_grad_array_gpu, 1),
                                                   size(term_col_cost_grad_array_gpu, 2),
                                                   0,
                                                   size(term_col_cost_grad_array_gpu, 4));
end

using Distributions
using LinearAlgebra

# Scene loader & Predictor parameters
scene_mode = "shiftedsynthetic";                                                           # "synthetic" or "data"
prediction_mode = "gaussian";                                                       # "gaussian" or "trajectron" or "oracle"
prediction_device = "cpu";                                                          # "cpu" or "cuda"
prediction_steps = 10;                                                              # number of steps to look ahead in the future
ado_pos_init_dict = Dict("PEDESTRIAN/1" => [0.0, -5.0],
                            "PEDESTRIAN/2" => [2.0, 5.0],
                            "PEDESTRIAN/3" => [-4.0, -2.0],
                            "PEDESTRIAN/4" => [4.0, 2.0],
                            "PEDESTRIAN/5" => [-4.0, -4.0],
                            "PEDESTRIAN/6" => [4.0, 4.0],
                            "PEDESTRIAN/7" => [5.0, -5.0],
                            "PEDESTRIAN/8" => [-5.0, 5.0],
                            "PEDESTRIAN/9" => [5.0, 5.0],
                            "PEDESTRIAN/10" => [-5.0, -5.0]);                            # initial ado positions [x, y] [m]
ado_vel_dict = Dict("PEDESTRIAN/1" => MvNormal([0.0, 1.0], Diagonal([0.1, 0.15])),
                            "PEDESTRIAN/2" => MvNormal([0.0, -1.0], Diagonal([0.1, 0.15])),
                            "PEDESTRIAN/3" => MvNormal([1.0, 0.0], Diagonal([0.15, 0.1])),
                            "PEDESTRIAN/4" => MvNormal([-1.0, 0.0], Diagonal([0.15, 0.1])),
                            "PEDESTRIAN/5" => MvNormal([1.0, 1.0], Diagonal([0.1, 0.1])),
                            "PEDESTRIAN/6" => MvNormal([-1.0, -1.0], Diagonal([0.1, 0.1])),
                            "PEDESTRIAN/7" => MvNormal([0.0, 1.0], Diagonal([0.1, 0.15])),
                            "PEDESTRIAN/8" => MvNormal([0.0, -1.0], Diagonal([0.1, 0.15])),
                            "PEDESTRIAN/9" => MvNormal([-1.0, 0.0], Diagonal([0.15, 0.1])),
                            "PEDESTRIAN/10" => MvNormal([1.0, 0.0], Diagonal([0.15, 0.1])));    # ado velocity distributions

ado_true_vel_dict = Dict("PEDESTRIAN/1" => [MvNormal([-0.1, 1.0], Diagonal([0.2, 0.15])), MvNormal([0.1, 1.0], Diagonal([0.2, 0.15]))],
                            "PEDESTRIAN/2" => [MvNormal([0.1, -1.0], Diagonal([0.2, 0.15])), MvNormal([-0.1, -1.0], Diagonal([0.2, 0.15]))],
                            "PEDESTRIAN/3" => [MvNormal([1.0, 0.1], Diagonal([0.15, 0.2])), MvNormal([1.0, -0.1], Diagonal([0.15, 0.2]))],
                            "PEDESTRIAN/4" => [MvNormal([-1.0, 0.1], Diagonal([0.15, 0.2])), MvNormal([-1.0, -0.1], Diagonal([0.15, 0.2]))],
                            "PEDESTRIAN/5" => [MvNormal([0.9, 0.9], Diagonal([0.3, 0.3])), MvNormal([1.1, 1.1], Diagonal([0.3, 0.3]))],
                            "PEDESTRIAN/6" => [MvNormal([-0.9, -0.9], Diagonal([0.3, 0.3])), MvNormal([-1.1, -1.1], Diagonal([0.3, 0.3]))],
                            "PEDESTRIAN/7" => [MvNormal([-0.1, 1.0], Diagonal([0.2, 0.15])), MvNormal([0.1, 1.0], Diagonal([0.2, 0.15]))],
                            "PEDESTRIAN/8" => [MvNormal([0.1, -1.0], Diagonal([0.2, 0.15])), MvNormal([-0.1, -1.0], Diagonal([0.2, 0.15]))],
                            "PEDESTRIAN/9" => [MvNormal([-1.0, 0.1], Diagonal([0.15, 0.2])), MvNormal([-1.0, -0.1], Diagonal([0.15, 0.2]))],
                            "PEDESTRIAN/10" => [MvNormal([1.0, 0.1], Diagonal([0.15, 0.2])), MvNormal([1.0, -0.1], Diagonal([0.15, 0.2]))]);    # true ado velocity distributions
dto = 0.4;                                                                          # observation update time interval [s]
prediction_rng_seed = 1;                                                            # random seed for prediction (and stochastic transition for "synthetic" scenes)
deterministic = false;                                                              # if true, a single, deterministic sample is drawn regardless of random seed. (num_samples = 1 is needed)
num_samples = 30;                                                                   # number of trajectory samples (per ado agent)
# Cost Parameters
include("params_drc_cost.jl")
# Control Parameters
include("params_drc_control.jl")
# Ego initial state
ego_pos_init_vec = [-5., 0.];                                                       # initial ego position [x, y] [m]
ego_pos_goal_vec = [5., 0.];                                                        # goal ego position [x, y] [m]
# Other parameters
pos_error_replan = 5.0;                                                             # position error for replanning target trajectory [m]
target_speed = 1.0;                                                                 # target speed [m/s]
sim_horizon = 10.0;                                                                 # simulation time horizon [s]

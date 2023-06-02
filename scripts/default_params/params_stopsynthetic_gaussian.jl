using Distributions
using LinearAlgebra

# Scene loader & Predictor parameters
scene_mode = "synthetic";                                                           # "synthetic" or "data"
prediction_mode = "stop_gaussian";                                                       # "gaussian" or "trajectron" or "oracle"
prediction_device = "cpu";                                                          # "cpu" or "cuda"
prediction_steps = 10;                                                              # number of steps to look ahead in the future
ado_pos_init_dict = Dict("PEDESTRIAN/1" => [0.0, -5.0],
                            "PEDESTRIAN/2" => [2.0, -5.0],
                            "PEDESTRIAN/3" => [-4.0, -2.0],
                            "PEDESTRIAN/4" => [4.0, 2.0],
                            "PEDESTRIAN/5" => [-4.0, -5.0],
                            "PEDESTRIAN/6" => [-5.0, -4.0],
                            "PEDESTRIAN/7" => [5.0, 3.0],
                            "PEDESTRIAN/8" => [-5.0, -3.0],
                            "PEDESTRIAN/9" => [5.0, 5.0],
                            "PEDESTRIAN/10" => [-5.0, -5.0]);                            # initial ado positions [x, y] [m]
ado_vel_dict = Dict("PEDESTRIAN/1" => [MvNormal([0.0, 1.0], Diagonal([0.1, 0.15])), MvNormal([0.0, 0.0], Diagonal([0.15, 0.15]))],
                            "PEDESTRIAN/2" => [MvNormal([0.0, 1.0], Diagonal([0.1, 0.15])), MvNormal([0.0, 0.0], Diagonal([0.15, 0.15]))],
                            "PEDESTRIAN/3" => [MvNormal([1.0, 0.0], Diagonal([0.15, 0.1])), MvNormal([0.0, 0.0], Diagonal([0.15, 0.15]))],
                            "PEDESTRIAN/4" => [MvNormal([-1.0, 0.0], Diagonal([0.15, 0.1])), MvNormal([0.0, 0.0], Diagonal([0.15, 0.15]))],
                            "PEDESTRIAN/5" => [MvNormal([1.0, 1.0], Diagonal([0.1, 0.1])), MvNormal([0.0, 0.0], Diagonal([0.15, 0.15]))],
                            "PEDESTRIAN/6" => [MvNormal([1.0, 1.0], Diagonal([0.1, 0.1])), MvNormal([0.0, 0.0], Diagonal([0.15, 0.15]))],
                            "PEDESTRIAN/7" => [MvNormal([-1.0, 0.0], Diagonal([0.1, 0.15])), MvNormal([0.0, 0.0], Diagonal([0.15, 0.15]))],
                            "PEDESTRIAN/8" => [MvNormal([1.0, 0.0], Diagonal([0.1, 0.15])), MvNormal([0.0, 0.0], Diagonal([0.15, 0.15]))],
                            "PEDESTRIAN/9" => [MvNormal([-1.0, 0.0], Diagonal([0.15, 0.1])), MvNormal([0.0, 0.0], Diagonal([0.15, 0.15]))],
                            "PEDESTRIAN/10" => [MvNormal([1.0, 0.0], Diagonal([0.15, 0.1])), MvNormal([0.0, 0.0], Diagonal([0.15, 0.15]))]);    # ado velocity distributions

dto = 0.4;                                                                          # observation update time interval [s]
prediction_rng_seed = 1;                                                            # random seed for prediction (and stochastic transition for "synthetic" scenes)
deterministic = false;                                                              # if true, a single, deterministic sample is drawn regardless of random seed. (num_samples = 1 is needed)
num_samples = 30;                                                                   # number of trajectory samples (per ado agent)
# Cost Parameters
include("params_cost.jl")
# Control Parameters
include("params_control.jl")
# Ego initial state
ego_pos_init_vec = [-5., 0.];                                                       # initial ego position [x, y] [m]
ego_pos_goal_vec = [5., 0.];                                                        # goal ego position [x, y] [m]
# Other parameters
pos_error_replan = 5.0;                                                             # position error for replanning target trajectory [m]
target_speed = 1.0;                                                                 # target speed [m/s]
sim_horizon = 10.0;                                                                 # simulation time horizon [s]

using Distributions
using LinearAlgebra

# Scene loader & Predictor parameters
scene_mode = "data";                                                                # "synthetic" or "data"
prediction_mode = "trajectron";                                                     # "gaussian" or "trajectron" or "oracle"
conf_file_name = "config.json"                                                      # trajectron config file name
test_data_name = "eth_test.pkl";                                                    # test data set name
test_scene_id = 0;                                                                  # test data id
start_time_idx = 905;                                                               # start time index in test data
incl_robot_node = false;                                                            # if true, robot node is created in trajectron
prediction_device = "cpu";                                                          # "cpu" or "cuda"
prediction_steps = 10;                                                              # number of steps to look ahead in the future
prediction_rng_seed = 1;                                                            # random seed for prediction (and stochastic transition for "synthetic" scenes)
deterministic = false;                                                              # if true, a single, deterministic sample is drawn regardless of random seed. (num_samples = 1 is needed)
num_samples = 30;                                                                   # number of trajectory samples (per ado agent)
use_robot_future = false;                                                           # if true, prediction is conditioned on robot's nominal control
# Cost Parameters
include("params_drc_cost.jl")
# Control Parameters
include("params_drc_control.jl")
# Ego initial state
ego_pos_init_vec = [5., 0.5] .+ [-5.263534, -5.314636];                             # initial ego position [x, y] [m]
ego_pos_goal_vec = [5., 8.9] .+ [-5.263534, -5.314636];                             # goal ego position [x, y] [m]
# Other parameters
pos_error_replan = 2.0;                                                             # position error for replanning target trajectory [m]
target_speed = 1.0;                                                                 # target speed [m/s]
sim_horizon = 10.0;      
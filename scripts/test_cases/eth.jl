test_data_name = "eth_test.pkl";                                                    # test data set name
test_scene_id = 0;                                                                  # test data id
start_time_idx = 905;                                                               # start time index in test data
ego_pos_init_vec = [5., 0.5] .+ [-5.263534, -5.314636];                             # initial ego position [x, y] [m]
ego_pos_goal_vec = [5., 8.9] .+ [-5.263534, -5.314636];                             # goal ego position [x, y] [m]
target_speed = 0.7;                                                                 # target speed [m/s]
sim_horizon = 10.0;                                                                 # simulation time horizon [s]

test_data_name = "hotel_test.pkl";                                                  # test data set name
test_scene_id = 0;                                                                  # test data id
start_time_idx = 401;                                                               # start time index in test data
ego_pos_init_vec = [-1.5, -8.5] .+ [-1.393743, 2.978962];                           # initial ego position [x, y] [m]
ego_pos_goal_vec = [3.5, 0.0]   .+ [-1.393743, 2.978962];                           # goal ego position [x, y] [m]
target_speed = 1.0;                                                                 # target speed [m/s]
sim_horizon = 10.0;                                                                 # simulation time horizon [s]

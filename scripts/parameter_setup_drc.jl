using Random

if !@isdefined constraint_time
    constraint_time = nothing
end

if scene_mode == "data" && prediction_mode == "trajectron"
    scene_param = TrajectronSceneParameter(conf_file_name, test_data_name,
                                           test_scene_id, start_time_idx,
                                           incl_robot_node);
    predictor_param = TrajectronPredictorParameter(prediction_steps,
                                                   num_samples,
                                                   use_robot_future,
                                                   deterministic,
                                                   prediction_rng_seed);
    cost_param = DRCCostParameter(ego_pos_goal_vec, Cep, Cu, β_pos, α_col, β_col, λ_col);
    cnt_param = DRCControlParameter(u_norm_max, tcalc, ego_pos_goal_vec, dtr, horizon, human_size,
                                        cem_init_mean, cem_init_cov, cem_init_num_samples,
                                        cem_init_num_elites, cem_init_alpha, cem_init_iterations, epsilon);

else
    @error "scene_mode: $(scene_mode) + prediction_mode: $(prediction_mode) is not supported!"
end

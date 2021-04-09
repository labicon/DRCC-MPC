if scene_mode == "data"
    scene_param = TrajectronSceneParameter(conf_file_name, test_data_name,
                                           test_scene_id, start_time_idx,
                                           incl_robot_node);
    cost_param = CostParameter(Cep, Cu, β_pos, α_col, β_col, λ_col, σ_risk);
    cnt_param = CrowdNavControlParameter(model_dir, env_config, policy_config,
                                         policy_name, ego_pos_goal_vec,
                                         tcalc, dtr, target_speed);
else
    @error @error "scene_mode: $(scene_mode) is not supported!"
end

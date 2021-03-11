using Random

if scene_mode == "synthetic"
    sim_rng_seed = prediction_rng_seed;
    sim_rng = MersenneTwister(sim_rng_seed);
    scene_param = SyntheticSceneParameter(sim_rng);
    cost_param = CostParameter(Cep, Cu, β_pos, α_col, β_col, λ_col, σ_risk);
    cnt_param = BICControlParameter(u_norm_max, tcalc, dtr, min_dist);

elseif scene_mode == "data"
    scene_param = TrajectronSceneParameter(conf_file_name, test_data_name,
                                           test_scene_id, start_time_idx,
                                           incl_robot_node);
    cost_param = CostParameter(Cep, Cu, β_pos, α_col, β_col, λ_col, σ_risk);
    cnt_param = BICControlParameter(u_norm_max, tcalc, dtr, min_dist);

else
    @error @error "scene_mode: $(scene_mode) is not supported!"
end

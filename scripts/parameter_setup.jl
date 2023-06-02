using Random

if !@isdefined constraint_time
    constraint_time = nothing
end

if scene_mode == "synthetic" && prediction_mode == "gaussian"
    rng = MersenneTwister(prediction_rng_seed);
    scene_param = SyntheticSceneParameter(rng);
    predictor_param = GaussianPredictorParameter(prediction_steps,
                                                 num_samples, deterministic, rng);
    cost_param = CostParameter(Cep, Cu, β_pos, α_col, β_col, λ_col, σ_risk);
    cnt_param = ControlParameter(u_norm_max, tcalc, dtexec, dtr,
                                 u_nominal_base, u_nominal_cand, nominal_search_depth,
                                 improvement_threshold, constraint_time=constraint_time)

elseif scene_mode == "synthetic" && prediction_mode == "stop_gaussian"
    rng = MersenneTwister(prediction_rng_seed);
    scene_param = StopSyntheticSceneParameter(rng);
    predictor_param = StopGaussianPredictorParameter(prediction_steps,
                                                 num_samples, deterministic, rng);
    cost_param = CostParameter(Cep, Cu, β_pos, α_col, β_col, λ_col, σ_risk);
    cnt_param = ControlParameter(u_norm_max, tcalc, dtexec, dtr,
                                 u_nominal_base, u_nominal_cand, nominal_search_depth,
                                 improvement_threshold, constraint_time=constraint_time)

elseif scene_mode == "shiftedsynthetic" && prediction_mode =="gaussian"
    rng = MersenneTwister(prediction_rng_seed);
    scene_param = ShiftedSyntheticSceneParameter(rng);
    predictor_param = GaussianPredictorParameter(prediction_steps,
                                                 num_samples, deterministic, rng);
    cost_param = CostParameter(Cep, Cu, β_pos, α_col, β_col, λ_col, σ_risk);
    cnt_param = ControlParameter(u_norm_max, tcalc, dtexec, dtr,
                                 u_nominal_base, u_nominal_cand, nominal_search_depth,
                                 improvement_threshold, constraint_time=constraint_time)                               

elseif scene_mode == "data" && prediction_mode == "gaussian"
    scene_param = TrajectronSceneParameter(conf_file_name, test_data_name,
                                           test_scene_id, start_time_idx,
                                           incl_robot_node);
    rng = MersenneTwister(prediction_rng_seed);
    predictor_param = GaussianPredictorParameter(prediction_steps,
                                                 num_samples, deterministic, rng);
    cost_param = CostParameter(Cep, Cu, β_pos, α_col, β_col, λ_col, σ_risk);
    cnt_param = ControlParameter(u_norm_max, tcalc, dtexec, dtr,
                                 u_nominal_base, u_nominal_cand, nominal_search_depth,
                                 improvement_threshold, constraint_time=constraint_time);

elseif scene_mode == "data" && prediction_mode == "trajectron"
    scene_param = TrajectronSceneParameter(conf_file_name, test_data_name,
                                           test_scene_id, start_time_idx,
                                           incl_robot_node);
    predictor_param = TrajectronPredictorParameter(prediction_steps,
                                                   num_samples,
                                                   use_robot_future,
                                                   deterministic,
                                                   prediction_rng_seed);
    cost_param = CostParameter(Cep, Cu, β_pos, α_col, β_col, λ_col, σ_risk);
    cnt_param = ControlParameter(u_norm_max, tcalc, dtexec, dtr,
                                 u_nominal_base, u_nominal_cand, nominal_search_depth,
                                 improvement_threshold, constraint_time=constraint_time);

elseif scene_mode == "data" && prediction_mode == "oracle"
    scene_param = TrajectronSceneParameter(conf_file_name, test_data_name,
                                           test_scene_id, start_time_idx,
                                           incl_robot_node);
    predictor_param = OraclePredictorParameter(prediction_steps);
    cost_param = CostParameter(Cep, Cu, β_pos, α_col, β_col, λ_col, σ_risk);
    cnt_param = ControlParameter(u_norm_max, tcalc, dtexec, dtr,
                                 u_nominal_base, u_nominal_cand, nominal_search_depth,
                                 improvement_threshold, constraint_time=constraint_time);

else
    @error "scene_mode: $(scene_mode) + prediction_mode: $(prediction_mode) is not supported!"
end

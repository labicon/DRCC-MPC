ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python")

using ArgParse
using Distributions
using LinearAlgebra
using FileIO
using JLD2
using Printf
using Random
using RiskSensitiveSAC
import RobotOS.Time

################ COMMAND LINE ARGUMENTS ################
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "params_file_name"
            help = "parameter file name in default_params (without .jl)"
            arg_type = String
            required = true
        "test_case_file_name_1"
            help = "a test case file name in test_cases (without .jl)"
            arg_type = String
            required = true
        "test_case_file_name_2"
            help = "a test case file name in test_cases (without .jl)"
            arg_type = String
        "test_case_file_name_3"
            help = "a test case file name in test_cases (without .jl)"
            arg_type = String
        "--sigma_risk"
            help = "σ_risk value (risk sensitivity parameter)"
            arg_type = Float64
        "--alpha_col"
            help = "α_col value (collision cost peak parameter)"
            arg_type = Float64
        "--lambda_col"
            help = "λ_col value (collision cost bandwidth parameter)"
            arg_type = Float64
        "--num_samples"
            help = "number of prediction samples (not used in bic controller or oracle predictor)"
            arg_type = Int64
        "--nominal_search_only"
            help = "use nominal search results only instead of rssac"
            action = :store_true
        "--nominal_base_only"
            help = "use nominal base only instead of rssac"
            action = :store_true
        "--deterministic_prediction"
            help = "perform deterministic prediction"
            action = :store_true
        "--future_conditional"
            help = "condition prediction on future nominal control"
            action = :store_true
        "--use_bic"
            help = "use buffered input cell controller instead of rssac"
            action = :store_true
        "--rng_seed"
            help = "rng seed for prediction (and simulation if in synthetic scene mode)"
            arg_type = Int64
        "--run_id"
            help = "run id"
            arg_type = Int64
            default = 1
        "--verbose"
            help = "verbose (true or false)"
            arg_type = Bool
            default = true
        "--show_log"
            help = "show log after evaluation is done"
            action = :store_true
        "--randomize_goal"
            help = "perturb goal location with unit Gaussian"
            action = :store_true
    end
    localARGS = (@isdefined newARGS) ? newARGS : ARGS
    return parse_args(localARGS, s)
end

################ SIMULATION SETUP ################
parsed_args = parse_commandline()
# check for consistency
if parsed_args["nominal_search_only"]
    @assert !parsed_args["nominal_base_only"]
end
if parsed_args["nominal_base_only"]
    @assert !parsed_args["nominal_search_only"]
end
if parsed_args["use_bic"]
    @assert !parsed_args["nominal_base_only"] &&
            !parsed_args["nominal_search_only"] &&
            !parsed_args["future_conditional"]
end

# load base parameters file
params_file_name = parsed_args["params_file_name"];
include("$(@__DIR__)/default_params/$(params_file_name).jl");
# load test case file(s)
test_case_file_name_1 = parsed_args["test_case_file_name_1"];
include("$(@__DIR__)/test_cases/$(test_case_file_name_1).jl");
if !isnothing(parsed_args["test_case_file_name_2"]);
    test_case_file_name_2 = parsed_args["test_case_file_name_2"];
    include("$(@__DIR__)/test_cases/$(test_case_file_name_2).jl");
end
if !isnothing(parsed_args["test_case_file_name_3"]);
    test_case_file_name_3 = parsed_args["test_case_file_name_3"];
    include("$(@__DIR__)/test_cases/$(test_case_file_name_3).jl");
end

# manually overwrite keyword arguments
if !isnothing(parsed_args["sigma_risk"])
    σ_risk = parsed_args["sigma_risk"];
end
if !isnothing(parsed_args["alpha_col"])
    α_col = parsed_args["alpha_col"];
end
if !isnothing(parsed_args["lambda_col"])
    λ_col = parsed_args["lambda_col"];
end
if !isnothing(parsed_args["num_samples"])
    num_samples = parsed_args["num_samples"]
end
if parsed_args["nominal_search_only"]
    dtexec = [0.0]
end
if parsed_args["deterministic_prediction"]
    deterministic = true
end
if parsed_args["future_conditional"]
    incl_robot_node = true
    use_robot_future = true
end
if !isnothing(parsed_args["rng_seed"])
    prediction_rng_seed = parsed_args["rng_seed"]
end
if parsed_args["randomize_goal"]
    goal_rng = MersenneTwister(prediction_rng_seed);
    d = MvNormal(zeros(2), Diagonal([1.0, 1.0]));
    ego_pos_goal_vec .+= rand(goal_rng, d)
end

# setup parameters
if parsed_args["use_bic"]
    include("$(@__DIR__)/parameter_setup_bic.jl");
    if scene_mode == "synthetic"
        scene_loader, controller, w_init, measurement_schedule, target_trajectory, target_speed, predictor =
        controller_setup(scene_param,
                         cost_param=cost_param,
                         cnt_param=cnt_param,
                         dtc=dtc,
                         dto=dto,
                         prediction_steps=prediction_steps,
                         num_samples=num_samples,
                         sim_rng=sim_rng,
                         ado_pos_init_dict=ado_pos_init_dict,
                         ado_vel_dict=ado_vel_dict,
                         ego_pos_init_vec=ego_pos_init_vec,
                         ego_vel_init_vec=nothing,
                         ego_pos_goal_vec=ego_pos_goal_vec,
                         target_speed=target_speed,
                         sim_horizon=sim_horizon,
                         verbose=parsed_args["verbose"]);

    elseif scene_mode == "data"
        scene_loader, controller, w_init, measurement_schedule, target_trajectory, target_speed =
        controller_setup(scene_param,
                         cost_param=cost_param,
                         cnt_param=cnt_param,
                         dtc=dtc,
                         prediction_steps=prediction_steps,
                         ego_pos_init_vec=ego_pos_init_vec,
                         ego_pos_goal_vec=ego_pos_goal_vec,
                         target_speed=target_speed,
                         sim_horizon=sim_horizon,
                         verbose=parsed_args["verbose"]);

    else
        @error @error "scene_mode: $(scene_mode) is not supported!"
    end
else
    include("$(@__DIR__)/parameter_setup.jl")
    if scene_mode == "synthetic" && prediction_mode == "gaussian"
        scene_loader, controller, w_init, measurement_schedule, target_trajectory, target_speed =
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
                         verbose=parsed_args["verbose"])

    elseif scene_mode == "data" && prediction_mode == "trajectron"
        if !@isdefined ado_id_to_replace
            ado_id_to_replace = nothing;
        end
        scene_loader, controller, w_init, measurement_schedule, target_trajectory, target_speed =
        controller_setup(scene_param,
                         predictor_param,
                         prediction_device=prediction_device,
                         cost_param=cost_param,
                         cnt_param=cnt_param,
                         dtc=dtc,
                         ego_pos_init_vec=ego_pos_init_vec,
                         ego_vel_init_vec=nothing,
                         ego_pos_goal_vec=ego_pos_goal_vec,
                         target_speed=target_speed,
                         sim_horizon=sim_horizon,
                         ado_id_to_replace=ado_id_to_replace,
                         verbose=parsed_args["verbose"])

    elseif scene_mode == "data" && prediction_mode == "oracle"
        scene_loader, controller, w_init, measurement_schedule, target_trajectory, target_speed =
        controller_setup(scene_param,
                         predictor_param,
                         prediction_device=prediction_device,
                         cost_param=cost_param,
                         cnt_param=cnt_param,
                         dtc=dtc,
                         ego_pos_init_vec=ego_pos_init_vec,
                         ego_vel_init_vec=nothing,
                         ego_pos_goal_vec=ego_pos_goal_vec,
                         target_speed=target_speed,
                         sim_horizon=sim_horizon,
                         verbose=parsed_args["verbose"])
    else
        @error "scene_mode: $(scene_mode) + prediction_mode: $(prediction_mode) is not supported!"
    end
end

#=
println("σ_risk:        $(controller.sim_param.cost_param.σ_risk)");
println("α_col:         $(controller.sim_param.cost_param.α_col)");
println("λ_col:         $(controller.sim_param.cost_param.λ_col)");
println("num_samples:   $(controller.sim_param.num_samples)");
println("dtexec:        $(controller.cnt_param.dtexec)");
try
    println("deterministic: $(controller.predictor.param.deterministic)")
catch e
end
try
    println("robot future:  $(controller.predictor.param.use_robot_future)")
catch e
end
try
    println("incl robot:    $(scene_loader.param.incl_robot_node)")
catch e
end
println("prediction_rng_seed:        $(prediction_rng_seed)");
=#

################ EVALUATION START ################
if !@isdefined predictor
    predictor = nothing;
end
if typeof(controller) == BICController
    nominal_control = nothing;
else
    nominal_control = parsed_args["nominal_base_only"]
end
if !@isdefined ado_id_removed
    ado_id_removed = nothing;
end
if isnothing(ego_pos_goal_vec)
    ego_pos_goal_vec = maximum(target_trajectory)[2]
end
println("Evaulation has started.")
result, ~, ~ = evaluate(scene_loader, controller, w_init, ego_pos_goal_vec,
                        target_speed, measurement_schedule, target_trajectory,
                        pos_error_replan, nominal_control=nominal_control,
                        ado_id_removed=ado_id_removed,
                        predictor=predictor);
if parsed_args["show_log"]
    display_log(result.log);
end
println("")
println("###### Simulation Result #######");
cnt_cost_str = @sprintf "Total Control Cost  : %10.4f" result.total_cnt_cost;
pos_cost_str = @sprintf "Total Position Cost : %10.4f" result.total_pos_cost;
col_cost_str = @sprintf "Total Collision Cost: %10.4f" result.total_col_cost;
min_dist_val = minimum([minimum(vcat([norm(get_position(w.e_state) - ap) for ap in values(w.ap_dict)], Inf))
                      for w in result.w_history]);
init_dist_goal_val = norm(get_position(result.w_history[1].e_state) - ego_pos_goal_vec);
final_dist_goal_val = norm(get_position(result.w_history[end].e_state) - ego_pos_goal_vec);
relative_goal_dist_val = final_dist_goal_val/init_dist_goal_val;
min_dist_str =      @sprintf "Minimum E-A Dist [m]: %10.4f" min_dist_val;
rel_goal_dist_str = @sprintf "Rel Goal Dist       : %10.4f" relative_goal_dist_val;
println(cnt_cost_str);
println(pos_cost_str);
println(col_cost_str);
println(min_dist_str);
println(rel_goal_dist_str);

################ SAVE ################
save_dir = normpath(joinpath(@__DIR__, "..", "simulation_data",
                             parsed_args["params_file_name"]));

if (@isdefined test_case_file_name_2) && (@isdefined test_case_file_name_3)
    test_case_name = "$(test_case_file_name_1)_$(test_case_file_name_2)_$(test_case_file_name_3)";
elseif @isdefined test_case_file_name_2
    test_case_name = "$(test_case_file_name_1)_$(test_case_file_name_2)";
else
    test_case_name = test_case_file_name_1;;
end
save_dir = joinpath(save_dir, test_case_name);

if (@isdefined incl_robot_node) && incl_robot_node &&
   (@isdefined use_robot_future) && use_robot_future
    save_dir = joinpath(save_dir, "future_conditional")
else
    save_dir = joinpath(save_dir, "non_future_conditional")
end

if typeof(controller) == BICController
    save_dir = joinpath(save_dir, "deterministic")
elseif deterministic || typeof(controller.predictor) == OraclePredictor
    save_dir = joinpath(save_dir, "deterministic")
else
    num_samples_str = @sprintf "%03d" controller.sim_param.num_samples;
    save_dir = joinpath(save_dir, "stochastic", "samples_$(num_samples_str)")
end
if !isdir(save_dir)
    mkpath(save_dir)
end

alpha_str = @sprintf "%05.1f" controller.sim_param.cost_param.α_col;
lambda_str = @sprintf "%04.2f" controller.sim_param.cost_param.λ_col;
risk_str = @sprintf "%05.2f" controller.sim_param.cost_param.σ_risk;
file_name = "alpha_$(alpha_str)_lambda_$(lambda_str)_risk_$(risk_str)_";
run_id_str = @sprintf "%03d" parsed_args["run_id"];
file_name *= "run_$(run_id_str).jld2"

save(save_dir * "/$(file_name)", "result", result, "parsed_args", parsed_args);
println("Results saved to " * save_dir * "/$(file_name)");

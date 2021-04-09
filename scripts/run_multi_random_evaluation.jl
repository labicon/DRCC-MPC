ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python")

using ArgParse
using ProgressMeter
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
        "--use_crowd_nav"
            help = "use crowd nav controller instead of rssac"
            action = :store_true
        "--rng_seed"
            help = "rng seed for prediction (and simulation if in synthetic scene mode)"
            arg_type = Int64
        "--verbose"
            help = "verbose (true or false)"
            arg_type = Bool
            default = true
        "--show_log"
            help = "show log after evaluation is done"
            action = :store_true
        "--num_runs"
            help = "number of random simulation runs. default = 10"
            arg_type = Int64
            default = 10
        "--global_seed"
            help = "use global rng seed"
            action = :store_true
        "--randomize_goal"
            help = "perturb goal location with unit Gaussian"
            action = :store_true
    end
    return parse_args(s)
end


function main()
    parsed_args = parse_commandline()
    params_file_name = parsed_args["params_file_name"];
    test_case_file_name_1 = parsed_args["test_case_file_name_1"]
    test_case_file_name_2 = parsed_args["test_case_file_name_2"]
    test_case_file_name_3 = parsed_args["test_case_file_name_3"]
    sigma_risk = parsed_args["sigma_risk"]
    alpha_col = parsed_args["alpha_col"]
    lambda_col = parsed_args["lambda_col"]
    num_samples = parsed_args["num_samples"]
    nominal_search_only = parsed_args["nominal_search_only"]
    nominal_base_only = parsed_args["nominal_base_only"]
    deterministic_prediction = parsed_args["deterministic_prediction"]
    future_conditional = parsed_args["future_conditional"]
    use_bic = parsed_args["use_bic"]
    use_crowd_nav = parsed_args["use_crowd_nav"]
    verbose = parsed_args["verbose"]
    show_log = parsed_args["show_log"]
    randomize_goal = parsed_args["randomize_goal"]

    if parsed_args["global_seed"]
        rng = Random.GLOBAL_RNG
    else
        rng = MersenneTwister(593803);
    end
    seed_pool = Int[];

    @showprogress for run_id = 1:parsed_args["num_runs"]
        rng_seed = nothing;
        while true
            rng_seed_candidate = abs(rand(rng, Int32));
            if !in(rng_seed_candidate, seed_pool)
                rng_seed = rng_seed_candidate;
                push!(seed_pool, rng_seed);
                break;
            end
        end
        global newARGS = [string(params_file_name),
                          string(test_case_file_name_1)];
        if !isnothing(test_case_file_name_2)
            push!(newARGS, string(test_case_file_name_2))
        end
        if !isnothing(test_case_file_name_3)
            push!(newARGS, string(test_case_file_name_3))
        end
        push!(newARGS, "--run_id")
        push!(newARGS, string(run_id))
        push!(newARGS, "--rng_seed")
        push!(newARGS, string(rng_seed))
        if !isnothing(sigma_risk)
            push!(newARGS, "--sigma_risk")
            push!(newARGS, string(sigma_risk))
        end
        if !isnothing(alpha_col)
            push!(newARGS, "--alpha_col")
            push!(newARGS, string(alpha_col))
        end
        if !isnothing(lambda_col)
            push!(newARGS, "--lambda_col")
            push!(newARGS, string(lambda_col))
        end
        if !isnothing(num_samples)
            push!(newARGS, "--num_samples")
            push!(newARGS, string(num_samples))
        end
        if nominal_search_only
            push!(newARGS, "--nominal_search_only")
        end
        if nominal_base_only
            push!(newARGS, "--nominal_base_only")
        end
        if deterministic_prediction
            push!(newARGS, "--deterministic_prediction")
        end
        if future_conditional
            push!(newARGS, "--future_conditional")
        end
        if use_bic
            push!(newARGS, "--use_bic")
        end
        if use_crowd_nav
            push!(newARGS, "--use_crowd_nav")
        end
        if !verbose
            push!(newARGS, "--verbose")
            push!(newARGS, string(false))
        end
        if show_log
            push!(newARGS, "--show_log")
        end
        if randomize_goal
            push!(newARGS, "--randomize_goal")
        end
        try
            include("run_single_evaluation.jl")
        catch
            @warn "Run $(run_id) has errored. Skipping..."
            continue;
        end
        GC.gc();
    end
    # println(seed_pool)
end

main()

using Printf
using RobotOS
using PyCall

struct DRCEvaluationResult
    sim_param::SimulationParameter
    cnt_param::DRCControlParameter
    # scene_param::Union{TrajectronSceneParameter, SyntheticSceneParameter} # error for JLD2?
    # predictor_param::Union{TrajectronPredictorParameter, GaussianPredictorParameter} # error for JLD2?
    # nominal_control::Bool
    target_trajectory_history::Vector{Trajectory2D}
    measurement_time_history::Vector{Time}
    sim_horizon::Float64
    w_history::Vector{WorldState}
    u_history::Vector{Vector{Float64}}
    # u_schedule_history::Vector{OrderedDict{Time, Vector{Float64}}}
    # nominal_trajectory_history::Vector{Vector{Vector{Float64}}}
    prediction_dict_history::Vector{Union{Nothing, Dict{String, Array{Float64, 3}}}}
    # u_nominal_idx_history::Vector{Int64}
    total_cnt_cost::Float64
    total_pos_cost::Float64
    total_col_cost::Float64
    total_col::Int64
    log::Vector{Tuple{Time, String}}
end

function get_clipped_prediction_dict(prediction_dict::Dict{String, Array{Float64, 3}},
                                     num_samples::Int64,
                                     nominal_control_idx::Int64=1)
    clipped_preds_dict = Dict{String, Array{Float64, 3}}();
    for key in keys(prediction_dict)
        clipped_array =
            prediction_dict[key][num_samples*(nominal_control_idx - 1) + 1:num_samples*nominal_control_idx, :, :];
        clipped_preds_dict[key] = copy(clipped_array);
    end
    return clipped_preds_dict
end

function get_control(controller::DRCController,
                        w_init::WorldState,
                        measurement_schedule::Vector{Time},
                        target_trajectory::Trajectory2D,
                        predictor::nothing)
    prediction_dict_history = Vector{Union{Nothing, Dict{String, Array{Float64, 3}}}}();
    @assert istaskdone(controller.prediction_task);

    current_time = w_history[end].t;
    if current_time == measurement_schedule_time
        # Get new measurement
        # current robot position
        ado_positions = 

        # Schedule prediction task
        schedule_prediction!(controller, ado_inputs);
        w_history[end].ap_dict = convert_nodes_to_str(ado_positions);
        w_history[end].t_last_m = current_time;
    end

    # Scheduel control update
    schedule_control_update!(controller, w_history[end], target_trajectory, log=log);
    last_control_update_time = w_history[end].t;

    # Get control for current_time
    u = control!(controller, current_time, log=log);
    prediction_dict_history[end] = get_clipped_prediction_dict(controller.prediction_dict, 
                                                                controller.num_samples);

    return u
end

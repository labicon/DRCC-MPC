#///////////////////////////////////////
#// File Name: evaluation.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/02/25
#// Description: Evaluation helper functions for RiskSensitiveSAC
#///////////////////////////////////////

using Printf
using RobotOS
using PyCall

struct DRCEvaluationResult
    sim_param::SimulationParameter
    cnt_param::DRCControlParameter
    # scene_param::Union{TrajectronSceneParameter, SyntheticSceneParameter} # error for JLD2?
    # predictor_param::Union{TrajectronPredictorParameter, GaussianPredictorParameter} # error for JLD2?
    # nominal_control::Bool
    # target_trajectory_history::Vector{Trajectory2D}
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

function evaluate(scene_loader::SceneLoader,
                  controller::DRCController,
                  w_init::WorldState,
                  ego_pos_goal_vec::Vector{Float64},
                  # target_speed::Float64,
                  measurement_schedule::Vector{Time},
                #   target_trajectory::Trajectory2D,
                  # pos_error_replan::Float64;
                  # ado_inputs_init::Union{Nothing, Dict{T, Vector{Float64}} where T <: Union{PyObject, String}}=nothing, # only needed for CrowdNavController
                  # nominal_control::Union{Nothing, Bool}=nothing, # determines if nominal control is used in RSSAC controller
                  ado_id_removed::Union{Nothing, String}=nothing, # determines if ado_id_removed is removed from scenes with TrajectronSceneLoader
                  predictor::Union{Nothing, GaussianPredictor}=nothing) # needs to feed in GaussianPredictor if BICController is used with SyntheticSceneLoader
    # Assertions
    if typeof(scene_loader) != TrajectronSceneLoader
        @assert isnothing(ado_id_removed)
    end
    @assert isnothing(predictor)
    # Initial Cost Values
    total_position_cost = 0.0;
    total_control_cost = 0.0;
    total_collision_cost = 0.0;
    total_collision = 0;
    # Logs
    w_history = Vector{WorldState}();
    u_history = Vector{Vector{Float64}}();
    # target_trajectory_history = Vector{Trajectory2D}();
    # u_nominal_idx_history = Vector{Int64}();
    # nominal_trajectory_history = Vector{Vector{Vector{Float64}}}();
    log = Vector{Tuple{Time, String}}();
    if typeof(controller) == DRCController
        # u_schedule_history = Vector{OrderedDict{Time, Vector{Float64}}}();
        prediction_dict_history = Vector{Union{Nothing, Dict{String, Array{Float64, 3}}}}();
        @assert istaskdone(controller.prediction_task)
    end
    # Compute First Control
    schedule_control_update!(controller, w_init, log);
    last_control_update_time = w_init.t;
    wait(controller.control_update_task);
    push!(w_history, w_init);
    # push!(target_trajectory_history, target_trajectory);
    if typeof(controller) == DRCController
        push!(prediction_dict_history, get_clipped_prediction_dict(controller.prediction_dict,
                                                                   controller.sim_param.num_samples));
        prediction_dict_last = prediction_dict_history[end];
        #push!(nominal_trajectory_history, deepcopy(get_position.(controller.sim_result_tmp.e_state_array)));
        #push!(u_nominal_idx_history, controller.sim_result_tmp.u_nominal_idx);
    end
    sleep(1.0)
    global ado_positions = nothing;
    # global ado_inputs = ado_inputs_init;

    # Starting Simulation
    m_time_idx = 1;
    sim_end_time = measurement_schedule[end];
    prediction_horizon = controller.sim_param.dto*
                         controller.sim_param.prediction_steps;
    while w_history[end].t <= sim_end_time
        current_time = w_history[end].t;
        if current_time == measurement_schedule[m_time_idx]
            # Get new measurement
            msg_1 = "New measurement is obtained."
            push!(log, (current_time, msg_1))
            if typeof(scene_loader) == TrajectronSceneLoader
                ado_inputs = fetch_ado_positions!(scene_loader, return_full_state=true);
                ado_positions = reduce_to_positions(ado_inputs);
                if !isnothing(ado_id_removed)
                    key_to_remove = nothing
                    for key in keys(ado_positions)
                        if pybuiltin("str")(key) == ado_id_removed
                            key_to_remove = key
                        end
                    end
                    delete!(ado_positions, key_to_remove)
                    delete!(ado_inputs, key_to_remove)
                end
            elseif typeof(scene_loader) == SyntheticSceneLoader
                ado_positions = fetch_ado_positions!(scene_loader, controller.prediction_dict);
            end
            # Starting timer to keep track of computation time
            process_start_time = time();
            if current_time < sim_end_time
                if typeof(controller) == DRCController
                    # Schedule prediction
                    previous_ado_pos_dict = deepcopy(w_history[end].ap_dict);
                    msg_2 = "New prediction is scheduled."
                    push!(log, (current_time, msg_2))
                    if typeof(controller.predictor) == TrajectronPredictor &&
                            controller.predictor.param.use_robot_future
                        schedule_prediction!(controller, ado_inputs, previous_ado_pos_dict,
                                                w_history[end].e_state);
                    elseif typeof(controller.predictor) == TrajectronPredictor
                        schedule_prediction!(controller, ado_inputs);
                    else
                        schedule_prediction!(controller, ado_positions, previous_ado_pos_dict);
                    end
                    prediction_dict_history[end] = get_clipped_prediction_dict(controller.prediction_dict,
                                                                                controller.sim_param.num_samples);
                    wait(controller.prediction_task);
                end
            end
            w_history[end].ap_dict = convert_nodes_to_str(ado_positions);
            w_history[end].t_last_m = current_time;
            m_time_idx += 1;
        else
            # No new measurement
            # Starting timer to keep track of computation time
            process_start_time = time();
        end

        # Update target trajectory if necessary
        pos_error = get_position(w_history[end].e_state) -
                    get_position(target_trajectory, w_history[end].t);
        if norm(pos_error) > pos_error_replan
            msg = "Position deviation: $(round(norm(pos_error), digits=4)). Target Trajectory is replanned."
            push!(log, (current_time, msg));
            target_trajectory = get_nominal_trajectory(w_history[end].e_state,
                                                       ego_pos_goal_vec,
                                                       target_speed,
                                                       to_sec(sim_end_time - w_init.t),
                                                       prediction_horizon);
        end

        # Proceed further
        if current_time < sim_end_time
            if to_sec(current_time) ≈ to_sec(last_control_update_time) + controller.cnt_param.dtr;
                # Schedule control update
                schedule_control_update!(controller, w_history[end], log);
                last_control_update_time = w_history[end].t
                wait(controller.control_update_task);
            end
            # Get control for current_time
            u = control!(controller, current_time, log)
            prediction_dict_history[end] = get_clipped_prediction_dict(controller.prediction_dict,
                                                                        controller.sim_param.num_samples);

            # Stop timer and measure computation time so far in this iteration.
            elapsed = time() - process_start_time;

            # Accumulate cost
            total_position_cost +=
                instant_position_cost(w_history[end].e_state, controller.sim_param.cost_param)*
                controller.sim_param.dtc;
            total_control_cost +=
                instant_control_cost(u, controller.sim_param.cost_param)*
                controller.sim_param.dtc;
            for ap in values(w_history[end].ap_dict)
                total_collision_cost +=
                    instant_collision_cost(w_history[end].e_state, ap,
                                           controller.sim_param.cost_param)*
                    controller.sim_param.dtc;
                total_collision += check_collision(w_history[end].e_state, ap, controller.sim_param.cost_param);
            end

            # Ego transition for next timestep
            e_state_new = transition(w_history[end].e_state, u, controller.sim_param.dtc);
            w_new = WorldState(e_state_new, deepcopy(w_history[end].ap_dict),
                               w_history[end].t_last_m);

            push!(u_history, u);
            push!(w_history, w_new);
            push!(prediction_dict_history, get_clipped_prediction_dict(controller.prediction_dict,
                                                                        controller.sim_param.num_samples));

            if controller.sim_param.dtc > elapsed
                # sleep for the remaining time.
                sleep(controller.sim_param.dtc - elapsed);
            else
                t = @sprintf "Time %.2f" round(to_sec(current_time), digits=5)
                @warn "$(t) [sec]: This evaluation iteration took $(round(elapsed, digits=3)) [sec], which exceeds dtc."
            end
        else
            # End simulation
            break;
        end
    end

    # Add Terminal Cost
    total_position_cost +=
        terminal_position_cost(w_history[end].e_state, controller.sim_param.cost_param);
    for ap in values(w_history[end].ap_dict)
        total_collision_cost +=
            terminal_collision_cost(w_history[end].e_state, ap,
                                    controller.sim_param.cost_param);
        total_collision += check_collision(w_history[end].e_state, ap, controller.sim_param.cost_param);
    end

    # Finish All the Remaining Tasks
    if typeof(controller) == DRCController && !isnothing(controller.prediction_task)
        wait(controller.prediction_task)
    end
    if !isnothing(controller.control_update_task)
        wait(controller.control_update_task)
    end

    # Return Simulation Results
    if typeof(controller) == DRCController
        # Only Save Unique Prediction_Dicts for Saving Disk Space
        jj = 1;
        for ii = 2:length(prediction_dict_history)
            if prediction_dict_history[ii] == prediction_dict_history[jj]
                prediction_dict_history[ii] = nothing;
            else
                jj = ii
            end
        end

        eval_result =
        DRCEvaluationResult(controller.sim_param, controller.cnt_param,
                         # scene_loader.param, controller.predictor.param,
                         measurement_schedule, to_sec(sim_end_time - w_init.t),
                         w_history, u_history, 
                         prediction_dict_history,
                         total_control_cost, total_position_cost, total_collision_cost, total_collision, log);
    end

    return eval_result, controller, ado_positions
end

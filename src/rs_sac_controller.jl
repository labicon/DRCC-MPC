#///////////////////////////////////////
#// File Name: rs_sac_controller.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/01/29
#// Description: Risk Sensitive SAC Controller
#///////////////////////////////////////

using DataStructures
using LinearAlgebra
using Printf
using Random
using RobotOS
import Convex: Variable, norm, quadform, minimize, dot, solve!
#using SCS
using ECOS

struct ControlParameter <: Parameter
    eamax::Float64  # Maximum abolute value of acceleration
    tcalc::Float64  # Allocated control computation time
    dtexec::Vector{Float64} # Control insertion duration candidates
    dtr::Float64    # Replanning time interval
    u_nominal_base::Vector{Float64} # baseline nominal control
    u_nominal_cand::Vector{Vector{Float64}} # Array of nominal control candidates (including u_nominal_base)
    nominal_search_depth::Int64
    improvement_threshold::Float64
    constraint_time::Union{Nothing, Float64}

    function ControlParameter(eamax::Float64, tcalc::Float64, dtexec::Vector{Float64},
                              dtr::Float64, u_nominal_base::Vector{Float64},
                              u_nominal_cand::Vector{Vector{Float64}},
                              nominal_search_depth::Int64,
                              improvement_threshold::Float64=0.0;
                              constraint_time::Union{Nothing, Float64}=nothing)
        @assert all(norm.(u_nominal_cand) .<= eamax)
        @assert in(u_nominal_base, u_nominal_cand)
        @assert tcalc <= dtr
        @assert nominal_search_depth >= 1
        @assert improvement_threshold >= 0.0
        @assert isnothing(constraint_time) || constraint_time > 0.0
        return new(eamax, tcalc, dtexec, dtr, u_nominal_base, u_nominal_cand,
                   nominal_search_depth, improvement_threshold,
                   constraint_time)
    end
end

mutable struct RSSACController
    sim_param::SimulationParameter
    cnt_param::ControlParameter
    predictor::Predictor

    u_schedule::OrderedDict{Time, Vector{Float64}}
    prediction_dict::Union{Nothing, Dict{String, Array{Float64, 3}}}
    sim_result::Union{Nothing, SimulationResult}
    tcalc_actual::Union{Nothing, Float64}
    #u_init_time::Union{Nothing, Time}
    #u_last_time::Union{Nothing, Time}
    #u_value::Union{Nothing, Vector{Float64}}

    prediction_task::Union{Nothing, Task}
    control_update_task::Union{Nothing, Task}

    u_schedule_tmp::Union{Nothing, OrderedDict{Time, Vector{Float64}}}
    prediction_dict_tmp::Union{Nothing, Dict{String, Array{Float64, 3}}}
    sim_result_tmp::Union{Nothing, SimulationResult}
    tcalc_actual_tmp::Union{Nothing, Float64}
    #u_init_time_tmp::Union{Nothing, Time}
    #u_last_time_tmp::Union{Nothing, Time}
    #u_value_tmp::Union{Nothing, Vector{Float64}}
end

function RSSACController(predictor::Predictor,
                         u_schedule::OrderedDict{Time, Vector{Float64}},
                         sim_param::SimulationParameter,
                         cnt_param::ControlParameter)
    return RSSACController(sim_param, cnt_param, predictor, u_schedule,
                           nothing, nothing, nothing, #nothing, nothing, nothing,
                           nothing, nothing,
                           nothing, nothing, nothing, nothing)#, nothing, nothing,
                           #nothing)
end

# # main control functions below
function control!(controller::RSSACController,
                  current_time::Time;
                  nominal_control::Bool=false,
                  log::Union{Nothing, Vector{Tuple{Time, String}}}=nothing)
    if !isnothing(controller.prediction_task) &&
        istaskdone(controller.prediction_task)
        if !isnothing(log)
            msg = "New prediction is available to the controller."
            push!(log, (current_time, msg))
        end
        controller.prediction_dict = copy(controller.prediction_dict_tmp);
        controller.prediction_task = nothing;
    end

    if !isnothing(controller.control_update_task) &&
        !isnothing(controller.u_schedule_tmp) &&
        istaskdone(controller.control_update_task)
        if !isnothing(log)
            msg = "New SAC control is available to the controller."
            push!(log, (current_time, msg))
        end
        controller.sim_result = deepcopy(controller.sim_result_tmp);
        controller.tcalc_actual = controller.tcalc_actual_tmp;
        #controller.u_init_time = controller.u_init_time_tmp;
        #controller.u_last_time = controller.u_last_time_tmp;
        #controller.u_value = copy(controller.u_value_tmp);
        controller.control_update_task = nothing;
        # update u_schedule as well.
        #=
        if controller.u_init_time < current_time
            if controller.u_last_time < current_time
                # This means that the sac control update was not performed in tcalc time.
                # In this case, just skip this part so the controller uses the nominal control.
                t = @sprintf "Time %.2f" round(to_sec(current_time), digits=5)
                @warn "$(t) [sec]: SAC control command is too old. Ignoring this SAC control."
            else
                # Rewrite u_init_time to make sure it is not in the past.
                controller.u_init_time = current_time;
            end
        end
        =#
        for time in keys(controller.u_schedule_tmp)
            if  !haskey(controller.u_schedule, time);
                continue;
            else
                controller.u_schedule[time] =
                    controller.u_schedule_tmp[time]
            end
        end
        controller.u_schedule_tmp = nothing;
    end

    if nominal_control
        pop!(controller.u_schedule, current_time);
        u = controller.cnt_param.u_nominal_base;
    else
        u = pop!(controller.u_schedule, current_time);
    end
    if !isnothing(log)
        msg = "Control: $(u) is applied to the system."
        push!(log, (current_time, msg))
    end

    # Insert new schedule
    new_control_time = collect(keys(controller.u_schedule))[end] +
                       Duration(controller.sim_param.dtc);
    @assert !haskey(controller.u_schedule, new_control_time)
    push!(controller.u_schedule, new_control_time => controller.cnt_param.u_nominal_base)

    return u
end

function adjust_old_prediction!(controller::RSSACController,
                                previous_ado_pos_dict::Dict{String, Vector{Float64}},
                                latest_ado_pos_dict::Dict{String, Vector{Float64}})
    @assert keys(controller.prediction_dict) == keys(previous_ado_pos_dict)

    ado_agents_removed = setdiff(keys(previous_ado_pos_dict),
                                 keys(latest_ado_pos_dict));
    ado_agents_added = setdiff(keys(latest_ado_pos_dict),
                               keys(previous_ado_pos_dict));
    keys_list = collect(keys(controller.prediction_dict))
    for key in keys_list
        if in(key, ado_agents_removed)
            # If this ado_agent is removed in latest_ado_pos_dict, then remove
            # it from prediction_dict.
            pop!(controller.prediction_dict, key)
        else
            if typeof(controller.predictor) != OraclePredictor # skip this step for OraclePredictor as predictions are perfect.
                # If this ado_agent is existent in both previous and latest
                # ado_pos_dict, then reuse and adjust the previous prediction.
                (diff_x, diff_y) = latest_ado_pos_dict[key] -
                                   previous_ado_pos_dict[key];
                controller.prediction_dict[key][:, :, 1] .+= diff_x;
                controller.prediction_dict[key][:, :, 2] .+= diff_y;
            end
        end
    end
    for key in ado_agents_added
        # Treat this newly added ado_agent as a static obstacle
        # (until the new prediction becomes available to the controller)
        num_controls = length(controller.cnt_param.u_nominal_cand)^controller.cnt_param.nominal_search_depth
        controller.prediction_dict[key] =
            zeros(controller.sim_param.num_samples*num_controls,
                  controller.sim_param.prediction_steps, 2);
        controller.prediction_dict[key][:, :, 1] .+= latest_ado_pos_dict[key][1];
        controller.prediction_dict[key][:, :, 2] .+= latest_ado_pos_dict[key][2];
    end
    @assert keys(controller.prediction_dict) == keys(latest_ado_pos_dict)
end

function schedule_prediction!(controller::RSSACController,
                              ado_pos_dict::Dict,
                              previous_ado_pos_dict::Union{Nothing, Dict{String, Vector{Float64}}}=nothing,
                              e_init::Union{Nothing, RobotState}=nothing);
    if !isnothing(previous_ado_pos_dict)
        adjust_old_prediction!(controller, previous_ado_pos_dict,
                               convert_nodes_to_str(reduce_to_positions(ado_pos_dict)));
    end
    if typeof(controller.predictor) == TrajectronPredictor
        controller.prediction_task = @task begin
            if controller.predictor.param.use_robot_future
                @assert !isnothing(e_init)
                robot_present_and_future=
                    get_robot_present_and_future(e_init,
                                                 controller.u_schedule,
                                                 controller.sim_param,
                                                 controller.cnt_param);

            else
                robot_present_and_future = nothing;
            end
            controller.prediction_dict_tmp =
                sample_future_ado_positions!(controller.predictor, ado_pos_dict,
                                             robot_present_and_future);
            if !controller.predictor.param.use_robot_future
                num_controls = length(controller.cnt_param.u_nominal_cand)^controller.cnt_param.nominal_search_depth;
                # each value has to be (num_samples*num_controls, prediction_steps, 2) array
                for key in keys(controller.prediction_dict_tmp)
                    controller.prediction_dict_tmp[key] =
                        repeat(controller.prediction_dict_tmp[key],
                               outer=(num_controls, 1, 1));
                end
            end
            # GC.gc();
        end
    elseif typeof(controller.predictor) == OraclePredictor
        controller.prediction_task = @task begin
            controller.prediction_dict_tmp =
                sample_future_ado_positions!(controller.predictor, ado_pos_dict);
            num_controls = length(controller.cnt_param.u_nominal_cand)^controller.cnt_param.nominal_search_depth;
            # each value has to be (num_samples*num_controls, prediction_steps, 2) array
            for key in keys(controller.prediction_dict_tmp)
                controller.prediction_dict_tmp[key] =
                    repeat(controller.prediction_dict_tmp[key],
                           outer=(num_controls, 1, 1));
            end
            # GC.gc();
        end
    elseif typeof(controller.predictor) == GaussianPredictor
        controller.prediction_task = @task begin
            controller.prediction_dict_tmp =
                sample_future_ado_positions!(controller.predictor,
                                             convert_nodes_to_str(ado_pos_dict));
            num_controls = length(controller.cnt_param.u_nominal_cand)^controller.cnt_param.nominal_search_depth;
            # each value has to be (num_samples*num_controls, prediction_steps, 2) array
            for key in keys(controller.prediction_dict_tmp)
                controller.prediction_dict_tmp[key] =
                    repeat(controller.prediction_dict_tmp[key],
                           outer=(num_controls, 1, 1));
            end
        end
    else
        @error "Type of controller.predictor: $(typeof(controller.predictor)) not supported."
    end
    schedule(controller.prediction_task);
end

function schedule_control_update!(controller::RSSACController,
                                  w_init::WorldState,
                                  target_trajectory::Trajectory2D;
                                  log::Union{Nothing, Vector{Tuple{Time, String}}}=nothing)
    if !isnothing(controller.prediction_task) &&
        istaskdone(controller.prediction_task)
        if !isnothing(log)
            msg = "New prediction is available to the controller."
            push!(log, (w_init.t, msg))
        end
        controller.prediction_dict = copy(controller.prediction_dict_tmp);
        controller.prediction_task = nothing;
    end
    if !isnothing(log)
        msg = "New SAC control computation is scheduled."
        push!(log, (w_init.t, msg))
    end
    u_schedule_copied = copy(controller.u_schedule);
    controller.control_update_task = @task begin
        controller.tcalc_actual_tmp,
        controller.u_schedule_tmp, controller.sim_result_tmp =
            sac_control_update(w_init, u_schedule_copied,
                               target_trajectory,
                               controller.prediction_dict,
                               controller.sim_param, controller.cnt_param);
    end
    schedule(controller.control_update_task);
end


# # Helper functions below
# convert u_array (array) to u_schedule (ordereddict)
@inline function convert_to_schedule(t_init::Time,
                                     u_array::Vector{Vector{Float64}},
                                     sim_param::SimulationParameter)
    u_schedule = OrderedDict{Time, Vector{Float64}}();
    plan_horizon = length(u_array)*sim_param.dtc;
    t = t_init;
    for ii = 1:length(u_array)
        u_schedule[t] = u_array[ii];
        t += Duration(sim_param.dtc);
    end
    return u_schedule
end

@inline function get_nominal_u_arrays(u_schedule::OrderedDict{Time, Vector{Float64}},
                                      modification_init_time::Time,
                                      modification_final_time::Time,
                                      cnt_param::ControlParameter,
                                      dto::Float64)
    @assert issorted(u_schedule);
    ordered_times = collect(keys(u_schedule))

    num_controls = length(cnt_param.u_nominal_cand);
    u_arrays = Vector{Vector{Vector{Float64}}}(undef, num_controls^cnt_param.nominal_search_depth);

    @assert cnt_param.u_nominal_cand[1] == [0.0, 0.0];
    u_arrays[1] = copy(collect(values(u_schedule)));
    if cnt_param.dtexec == [0.0] # SAC is not used, so set this nominal to zero
        fill!(u_arrays[1], [0.0, 0.0])
    end

    for ii = 2:length(u_arrays)
        @inbounds u_arrays[ii] = copy(collect(values(u_schedule)));
        cnt_idx_array = reverse(digits(ii - 1, base=num_controls, pad=cnt_param.nominal_search_depth) .+ 1);
        mod_init_time_tmp = modification_init_time;
        mod_final_time_tmp = modification_final_time;
        for idx in cnt_idx_array
            init_idx = findfirst(x -> x == mod_init_time_tmp, ordered_times);
            final_idx = findfirst(x -> x == mod_final_time_tmp, ordered_times);
            if !(isnothing(init_idx) || isnothing(final_idx))
                for jj = init_idx:final_idx
                    @inbounds u_arrays[ii][jj] = copy(cnt_param.u_nominal_cand[idx]);
                end
                mod_init_time_tmp += Duration(dto);
                mod_final_time_tmp += Duration(dto);
            end
        end
    end
    return u_arrays
end


@inline function get_nominal_u_arrays(u_schedule::OrderedDict{Time, Vector{Float64}},
                                      sim_param::SimulationParameter,
                                      cnt_param::ControlParameter)
    # make sure nominal control modification only happens after tcalc [s]
    # to not override the sac control from previous iteration.
    curr_time = minimum(keys(u_schedule));
    u_nominal_mod_init_time = curr_time + Duration(cnt_param.tcalc);
    u_nominal_mod_final_time = u_nominal_mod_init_time +
                               Duration(sim_param.dto) -
                               Duration(sim_param.dtc);
    return get_nominal_u_arrays(u_schedule, u_nominal_mod_init_time,
                                u_nominal_mod_final_time, cnt_param,
                                sim_param.dto)
end

# for Trajectron robot-future-conditoinal models
@inline function get_robot_present_and_future(e_init::RobotState,
                                      u_schedule::OrderedDict{Time, Vector{Float64}},
                                      sim_param::SimulationParameter,
                                      cnt_param::ControlParameter)
    # compute u_arrays from u_schedule and nominal control candidates
    u_arrays = get_nominal_u_arrays(u_schedule, sim_param, cnt_param);
    # process u_arrays
    u_array_gpu = cu(process_u_arrays(u_arrays));
    # ego state simulation
    ex_array_gpu = simulate_forward(e_init, u_array_gpu, sim_param)
    slice_factor = Int64(round(sim_param.dto/sim_param.dtc, digits=5));
    ex_array_cpu = collect(ex_array_gpu[:, 1:slice_factor:end, :])
    @assert size(ex_array_cpu, 2) == sim_param.prediction_steps + 1

    # (num_controls, 1 + prediction_steps, 6) array where the last dimension
    # is [pos_x, pos_y, vel_x, vel_y, acc_x, acc_y]. accelerations are computed
    # by finite-differencing velocity snapshots.
    robot_present_and_future = Array{Float64, 3}(undef, size(ex_array_cpu, 1),
                                                        size(ex_array_cpu, 2),
                                                        6);
    robot_present_and_future[:, :, 1:4] = ex_array_cpu;
    acc = diff(ex_array_cpu[:, :, 3:4], dims=2)./sim_param.dto
    robot_present_and_future[:, 1:end-1, 5:6] = acc;
    # pad last acceleration
    robot_present_and_future[:, end, 5:6] = robot_present_and_future[:, end-1, 5:6]

    return robot_present_and_future
end

@inline function get_control_coeffs(sampled_total_costs::Vector{Float64},
                                    e_state_array::Vector{RobotState},
                                    e_costate_array::Array{Float64, 3},
                                    valid_timesteps::Int64,
                                    sim_param::SimulationParameter,
                                    cnt_param::ControlParameter)
    # a trick to prevent underflow or overflow in computing sum of exponentials
    total_cost_extreme = nothing;
    if sim_param.cost_param.σ_risk >= 0.0
        total_cost_extreme = maximum(sampled_total_costs);
    else
        total_cost_extreme = minimum(sampled_total_costs);
    end
    coeffs = sim_param.cost_param.σ_risk.*(sampled_total_costs .- total_cost_extreme);

    # denominator
    den = sum(exp.(coeffs), dims=1);
    num = sum(exp.(coeffs).*
              [e_costate_array[:, ii, jj] for jj = 1:sim_param.num_samples, ii = 1:valid_timesteps],
              dims=1);

    coeff_array = transpose.(num./den);
    coeff_array .*= reshape(transition_control_coeff.(e_state_array[1:valid_timesteps]),
                           1, valid_timesteps);
    coeff_matrix = hcat(transpose.(coeff_array)...)

    return coeff_matrix
end


@inline function get_control_coeffs(sim_result::SimulationResult,
                            sim_param::SimulationParameter,
                            cnt_param::ControlParameter)
    # timesteps between current and (first measurement time + tcalc)
    #valid_timesteps = Int64((cnt_param.dtr + cnt_param.tcalc)/sim_param.dtc);
    sim_horizon = sim_param.prediction_steps*sim_param.dto;
    valid_timesteps = Int64(round(sim_horizon/sim_param.dtc, digits=5));
    coeff_matrix = get_control_coeffs(sim_result.sampled_total_costs,
                                      sim_result.e_state_array,
                                      sim_result.e_costate_array,
                                      valid_timesteps, sim_param, cnt_param);

    if !isnothing(cnt_param.constraint_time)
        valid_timesteps_constraint = size(sim_result.e_costate_array_constraint, 2) - 1;
        @assert valid_timesteps_constraint <= valid_timesteps
        coeff_matrix_constraint =
            get_control_coeffs(sim_result.sampled_total_costs_constraint,
                               sim_result.e_state_array,
                               sim_result.e_costate_array_constraint,
                               valid_timesteps_constraint, sim_param, cnt_param)
    else
        coeff_matrix_constraint = nothing
    end

    return coeff_matrix, coeff_matrix_constraint
end

struct ControlSchedule
    u::Vector{Float64}
    cost::Float64
    t::Time
end

@inline function solve_multi_qcqp(u_nominal_array::Vector{Vector{Float64}},
                                  coeff_matrix::Matrix{Float64},
                                  coeff_matrix_constraint::Matrix{Float64},
                                  sim_param::SimulationParameter,
                                  cnt_param::ControlParameter)
    u = Variable(2);
    cost_array = Vector{Float64}(undef, size(coeff_matrix_constraint, 2));
    u_array = Vector{Vector{Float64}}(undef, size(coeff_matrix_constraint, 2));
    Cu = sim_param.cost_param.Cu;
    for ii = 1:size(coeff_matrix_constraint, 2)
        coeff_vec = coeff_matrix[:, ii];
        u_nominal_vec = u_nominal_array[ii];
        coeff_vec_constraint = coeff_matrix_constraint[:, ii];
        if all(isapprox.(coeff_vec_constraint, 0.0, atol=1e-4))
            u_array[ii] = u_nominal_vec;
            cost_array[ii] = 0.0;
        else
            objective = 0.5*quadform(u, Cu) - dot(coeff_vec, (u - u_nominal_vec)) - 0.5*u_nominal_vec'*Cu*u_nominal_vec;
            problem = minimize(objective);
            problem.constraints += 0.5*quadform(u, Cu) - dot(coeff_vec_constraint, (u - u_nominal_vec)) -
                                   0.5*u_nominal_vec'*Cu*u_nominal_vec <= 0.0
            problem.constraints += norm(u, 2) <= cnt_param.eamax
            #solve!(problem, SCSSolver(verbose=false));
            solve!(problem, () -> ECOS.Optimizer(verbose=false));
            u_array[ii] = Float64[u.value[1], u.value[2]];
            cost_array[ii] = problem.optval;
        end
    end
    return u_array, cost_array
end

@inline function get_control_schedule(sim_result::SimulationResult,
                              u_nominal_array::Vector{Vector{Float64}},
                              coeff_matrix::Matrix{Float64},
                              sim_param::SimulationParameter,
                              cnt_param::ControlParameter,
                              coeff_matrix_constraint::Union{Nothing, Matrix{Float64}}=nothing)
    Cu = sim_param.cost_param.Cu;
    Cu_inv = inv(sim_param.cost_param.Cu);
    control_schedule_array = Vector{ControlSchedule}(undef,
                                                     size(coeff_matrix, 2));
    t_current = sim_result.e_state_array[1].t;

    if isnothing(coeff_matrix_constraint)
        time = t_current;
        for t = 1:size(coeff_matrix, 2)
            coeff_vec = coeff_matrix[:, t];
            u = -Cu_inv*coeff_vec;
            eamax = cnt_param.eamax;
            if norm(u) > eamax
                u = u./norm(u).*eamax; # This is valid as long as Cu is diagonal.
            end
            #cost = 1/2*u'*Cu*u + dot(u, coeff_vec);
            cost = 1/2*u'*Cu*u + dot((u - u_nominal_array[t]), coeff_vec) -
                   1/2*u_nominal_array[t]'*Cu*u_nominal_array[t];
            control_schedule_array[t] = ControlSchedule(u, cost, time);
            time += Duration(sim_param.dtc);
        end
    else
        u_array_constraint, cost_constraint =
            solve_multi_qcqp(u_nominal_array,
                             coeff_matrix,
                             coeff_matrix_constraint,
                             sim_param, cnt_param);
        time = t_current;
        for t = 1:size(coeff_matrix_constraint, 2)
            control_schedule_array[t] =
                ControlSchedule(u_array_constraint[t], cost_constraint[t], time);
                time += Duration(sim_param.dtc);
        end
        for t = size(coeff_matrix_constraint, 2)+1:size(coeff_matrix, 2)
            coeff_vec = coeff_matrix[:, t];
            u = -Cu_inv*coeff_vec;
            eamax = cnt_param.eamax;
            if norm(u) > eamax
                u = u./norm(u).*eamax; # This is valid as long as Cu is diagonal.
            end
            #cost = 1/2*u'*Cu*u + dot(u, coeff_vec);
            cost = 1/2*u'*Cu*u + dot((u - u_nominal_array[t]), coeff_vec) -
                   1/2*u_nominal_array[t]'*Cu*u_nominal_array[t];
            control_schedule_array[t] = ControlSchedule(u, cost, time);
                time += Duration(sim_param.dtc);
        end
    end
    return control_schedule_array
end

@inline function determine_control_time(
                                sim_result::SimulationResult,
                                control_schedule_array::Vector{ControlSchedule},
                                sim_param::SimulationParameter,
                                cnt_param::ControlParameter)
    # if dtexec is smaller than dtc, then use dtexec == dtc.
    if maximum(cnt_param.dtexec) < sim_param.dtc
        dtexec = sim_param.dtc;
    else
        dtexec = maximum(cnt_param.dtexec);
    end
    t_current = sim_result.e_state_array[1].t;
    t_allowed_min = t_current +
                    Duration(cnt_param.tcalc) +
                    Duration(dtexec);
    t_allowed_max = t_current +
                    Duration(sim_param.prediction_steps*sim_param.dto);
    if to_nsec(t_allowed_max - t_allowed_min) < 0.
        throw(ErrorException("t_min: $(to_sec(t_allowed_min)) > t_max:
                             $(to_sec(t_allowed_max)). There is no feasible
                             control insertion time."))
    end
    control_schedule_filtered =
        control_schedule_array[[t_allowed_min <= s.t <= t_allowed_max for s in
                                control_schedule_array]];
    control_chosen =
        control_schedule_filtered[findmin([s.cost for s in
                                  control_schedule_filtered])[2]];
    return control_chosen
end

@inline function perturbation_candidates(current_time::Time,
                                 control_chosen::ControlSchedule, u_nominal_array::Vector{Vector{Float64}},
                                 dtexec::Vector{Float64}, sim_param::SimulationParameter)
    u_arrays = Vector{Vector{Vector{Float64}}}();
    for dte in dtexec
        u_array = copy(u_nominal_array);
        final_idx = Int64(round(to_nsec(control_chosen.t - current_time)/
                                to_nsec(Duration(sim_param.dtc)),
                                digits=5));
        if dte <= sim_param.dtc
            if dte == 0.0
                push!(u_arrays, u_array)
            else
                u_array[final_idx] = control_chosen.u*(dte / sim_param.dtc);
                push!(u_arrays, u_array);
            end
        else
            start_time = control_chosen.t - Duration(dte);
            start_idx = Int64(round(to_nsec(start_time - current_time)/
                                    to_nsec(Duration(sim_param.dtc)),
                                    digits=5)) + 1
            for ii = start_idx:final_idx
                u_array[ii] = control_chosen.u
            end
            push!(u_arrays, u_array);
        end
    end
    return u_arrays
end

function get_perturbed_schedule(w_init::WorldState,
                                sim_result::SimulationResult,
                                u_perturb_arrays::Vector{Vector{Vector{Float64}}},
                                target_trajectory::Trajectory2D,
                                prediction_dict::Dict{String, Array{Float64, 3}},
                                sim_param::SimulationParameter,
                                cnt_param::ControlParameter)
    # process u_arrays
    u_array = process_u_arrays(u_perturb_arrays);
    u_array_gpu = cu(u_array);

    # ego state simulation
    ex_array_gpu = simulate_forward(w_init.e_state, u_array_gpu, sim_param);

    # process prediction_dict for further computation
    # note that each value in prediction_dict has to be (num_samples*num_controls, prediction_steps, 2) array
    num_controls = length(u_perturb_arrays)
    ap_array_gpu = repeat(sim_result.ap_array_gpu, outer=(num_controls, 1, 1, 1));
    measurement_schedule = sim_result.measurement_schedule;

    # ratio between dto and dtc
    expansion_factor = Int64(sim_param.dto/sim_param.dtc);
    # number of nominal control candidates
    num_controls = size(ex_array_gpu, 1);

    # total simulation timesteps
    total_timesteps = size(ex_array_gpu, 2);
    # timesteps between current and first measurement time
    init_timesteps =
        Int64(to_nsec(measurement_schedule[1] - w_init.t)/
              (sim_param.dtc*1e9));
    # timesteps between final measurement time and final e_state time
    remaining_timesteps = total_timesteps -
                          (init_timesteps + expansion_factor*(sim_param.prediction_steps - 1));
    # time_idx_ap_array
    time_idx_ap_array = Vector{Int64}(undef, total_timesteps);
    time_idx_ap_array[1:init_timesteps] .= 1;
    time_idx_ap_array[init_timesteps+1:end-remaining_timesteps] = repeat(2:1:sim_param.prediction_steps,
                                                                         inner=expansion_factor);
    time_idx_ap_array[end-remaining_timesteps+1:end] .= sim_param.prediction_steps + 1;

    # control_idx_ex_array
    control_idx_ex_array = repeat(1:1:num_controls, inner=sim_param.num_samples)

    time_idx_ap_array_gpu = CuArray{Int32}(time_idx_ap_array);
    control_idx_ex_array_gpu = CuArray{Int32}(control_idx_ex_array);

    # get target_pos array
    target_pos_array = get_target_pos_array(ex_array_gpu, w_init, target_trajectory, sim_param)
    target_pos_array_gpu = cu(target_pos_array);

    # compute costs
    cost_result = compute_costs(ex_array_gpu, u_array_gpu, ap_array_gpu,
                                time_idx_ap_array_gpu, control_idx_ex_array_gpu,
                                target_pos_array_gpu, sim_param.cost_param);

    # integrate costs
    total_cost_per_control_sample, risk_per_control = integrate_costs(cost_result, sim_param);
    if !isnothing(cnt_param.constraint_time)
        ~, risk_per_control_constraint = integrate_costs(cost_result, sim_param,
                                                         cnt_param.constraint_time);
        valid_indices = findall(risk_per_control_constraint .<= sim_result.risk_constraint);
        if isempty(valid_indices)
            time = @sprintf "Time %.2f" round(to_sec(w_init.t), digits=5)
            @info "$(time) [sec]: SAC did not find feasible improvement"
            nominal_schedule = convert_to_schedule(w_init.t, u_perturb_arrays[1], sim_param) # revert back to nominal control (assuming that 0.0 is the first element in dtexec)
            return sim_result.risk_val, nominal_schedule, 1
        end
    else
        valid_indices = 1:length(risk_per_control);
    end

    min_risk, control_idx = findmin(risk_per_control[valid_indices])
    if min_risk - sim_result.risk_val >= -cnt_param.improvement_threshold || length(valid_indices) == 1 # revert back to nominal control (assuming that 0.0 is the first element in dtexec)
        # because it's a stochastic algorithm you want to have some margin here.
        nominal_schedule = convert_to_schedule(w_init.t, u_perturb_arrays[1], sim_param)
        return sim_result.risk_val, nominal_schedule, 1
        #time = @sprintf "Time %.2f" round(to_sec(w_init.t), digits=5)
        #@info "$(time) [sec]: SAC did not improve risk"
    else
        best_schedule = convert_to_schedule(w_init.t, u_perturb_arrays[control_idx], sim_param)
        #time = @sprintf "Time %.2f" round(to_sec(w_init.t), digits=5)
        #@info "$(time) [sec]: SAC improved risk by $(min_risk - sim_result.risk_val)"
        return min_risk, best_schedule, control_idx
    end
end

@inline function sac_control_update(w_init::WorldState,
                            u_schedule::OrderedDict{Time, Vector{Float64}},
                            target_trajectory::Trajectory2D,
                            prediction_dict::Dict{String, Array{Float64, 3}},
                            sim_param::SimulationParameter,
                            cnt_param::ControlParameter)
    # dtexec has to be divisible by dtc, or smaller than dtc.
    @assert all([(to_nsec(Duration(dte)) %
            to_nsec(Duration(sim_param.dtc)) == 0) ||
            dte < sim_param.dtc for dte in cnt_param.dtexec])
    tcalc_actual =
        @elapsed u_arrays = get_nominal_u_arrays(u_schedule, sim_param,
                                                 cnt_param);
    tcalc_actual +=
        @elapsed sim_result, u_nominal_array =
            simulate(w_init, u_arrays, target_trajectory,
                     prediction_dict, sim_param, cnt_param.constraint_time);
    tcalc_actual +=
        @elapsed coeff_matrix, coeff_matrix_constraint =
            get_control_coeffs(sim_result, sim_param, cnt_param);
    tcalc_actual +=
        @elapsed control_schedule_array = get_control_schedule(sim_result,
                                                               u_nominal_array,
                                                               coeff_matrix,
                                                               sim_param,
                                                               cnt_param,
                                                               coeff_matrix_constraint);
    tcalc_actual +=
        @elapsed control_chosen =  determine_control_time(sim_result,
                                                          control_schedule_array,
                                                          sim_param, cnt_param);
    tcalc_actual +=
        @elapsed u_perturb_arrays = perturbation_candidates(w_init.t,
                                               control_chosen, u_nominal_array,
                                               cnt_param.dtexec, sim_param)
    tcalc_actual +=
        @elapsed min_risk, best_schedule, control_idx =
            get_perturbed_schedule(w_init, sim_result,
                                   u_perturb_arrays, target_trajectory,
                                   prediction_dict, sim_param, cnt_param);
    if tcalc_actual >= cnt_param.tcalc
        # In this case there is not much time left for the controller to insert control for dtexec seconds.
        # In simulation, throw the following warning message and override.
        time = @sprintf "Time %.2f" round(to_sec(w_init.t), digits=5)
        @warn "$(time) [sec]: SAC computation took $(round(tcalc_actual, digits=3)) [sec], which exceeds the maximum computation time allowed."
    end
    if min_risk >= sim_result.risk_val
    #    time = @sprintf "Time %.2f" round(to_sec(w_init.t), digits=5)
    #    @info "$(time) [sec]: SAC computation did not improve the best nominal risk value.
    else
    #    time = @sprintf "Time %.2f" round(to_sec(w_init.t), digits=5)
    #    @info "$(time) [sec]: SAC improved risk by $(min_risk - sim_result.risk_val)"
    end

    return tcalc_actual, best_schedule, sim_result
end

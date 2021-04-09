#///////////////////////////////////////
#// File Name: crowd_nav_controller.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/04/02
#// Description: Controller based on CrowdNav RL Policies
#///////////////////////////////////////

struct CrowdNavControlParameter <: Parameter
    model_dir::String # Directory of the trained policy
    env_config::String # environment config file name
    policy_config::String # policy config file name
    policy_name::String # policy name
    goal_pos::Vector{Float64} # [x, y] goal position
    tcalc::Float64 # Allocated control computation time
    dtr::Float64 # Replanning time interval
    target_speed::Float64 # target speed for robot
end

mutable struct CrowdNavController
    sim_param::SimulationParameter
    cnt_param::CrowdNavControlParameter

    rl_robot::PyObject

    tcalc_actual::Union{Nothing, Float64}
    u_value::Union{Nothing, Vector{Float64}}

    control_update_task::Union{Nothing, Task}

    tcalc_actual_tmp::Union{Nothing, Float64}
    u_value_tmp::Union{Nothing, Vector{Float64}}
end

function CrowdNavController(sim_param::SimulationParameter,
                            cnt_param::CrowdNavControlParameter)
    rl_robot = py"configure_rl_robot"(cnt_param.model_dir, cnt_param.env_config,
                                      cnt_param.policy_config, cnt_param.policy_name);
    @assert rl_robot.visible == false "RL Robot has to be invisible."
    @assert rl_robot.time_step == sim_param.dto "RL Robot has to use the same timestep as sim_param.dto."
    rl_robot.v_pref = cnt_param.target_speed;
    rl_robot.gx, rl_robot.gy = cnt_param.goal_pos[1], cnt_param.goal_pos[2];
    rl_robot.theta = 0.0
    return CrowdNavController(sim_param, cnt_param, rl_robot, nothing, nothing,
                              nothing, nothing, nothing);
end

function get_action!(controller::CrowdNavController,
                     e_init::RobotState,
                     ado_state_dict::Dict{T, Vector{Float64}} where T <: Union{PyObject, String})
    # update rl_robot state
    controller.rl_robot.set_position(copy(get_position(e_init)));
    controller.rl_robot.set_velocity(copy(get_velocity(e_init)));

    # ado_state_dict has to have velocity information
    ado_state_array = [];
    radius::Float64 = controller.rl_robot.radius
    if length(ado_state_dict) < 2 # crowdnav seems to need >1 ado agent
        vec = controller.cnt_param.goal_pos - get_position(e_init);
        action = vec./norm(vec).*controller.rl_robot.v_pref;
        return action
    else
        for key in keys(ado_state_dict)
            @assert length(ado_state_dict[key]) >= 4 "ado_state_dict has to have velocity information."
            px, py = ado_state_dict[key][1], ado_state_dict[key][2];
            vx, vy = ado_state_dict[key][3], ado_state_dict[key][4];
            observable_state = py"ObservableState"(px, py, vx, vy, radius);
            push!(ado_state_array, observable_state)
        end
        action = controller.rl_robot.act(ado_state_array);
        return action
    end
end

@inline function crowdnav_control_update!(controller::CrowdNavController,
                                          e_init::RobotState,
                                          ado_state_dict::Dict{T, Vector{Float64}} where T <: Union{PyObject, String})
    tcalc_actual =
        @elapsed u_vel = get_action!(controller, e_init, ado_state_dict);
    tcalc_actual +=
        @elapsed u_acc = (u_vel .- get_velocity(e_init)) ./ controller.sim_param.dto;
        #@elapsed u_acc = 2 * (u_vel .- get_velocity(e_init)) ./ controller.sim_param.dto;

    if tcalc_actual >= controller.cnt_param.tcalc
        # tcalc_actual has exceeded allowable computation time
        time = @sprintf "Time %.2f" round(to_sec(e_init.t), digits=5)
        @warn "$(time) [sec]: BIC computation took $(round(tcalc_actual, digits=3)) [sec], which exceeds the maximum computation time allowed."
    end

    return tcalc_actual, u_acc
end

function control!(controller::CrowdNavController,
                  current_time::Time;
                  log::Union{Nothing, Vector{Tuple{Time, String}}}=nothing)

    if !isnothing(controller.control_update_task) &&
        istaskdone(controller.control_update_task)
        if !isnothing(log)
            msg = "New CrowdNav control is available to the controller."
            push!(log, (current_time, msg))
        end
        controller.tcalc_actual = controller.tcalc_actual_tmp;
        controller.u_value = copy(controller.u_value_tmp);
        controller.control_update_task = nothing;
    end
    u = copy(controller.u_value);
    if !isnothing(log)
        msg = "Control: $(u) is applied to the system."
        push!(log, (current_time, msg))
    end
    return u
end

function schedule_control_update!(controller::CrowdNavController,
                                  e_init::RobotState,
                                  ado_state_dict::Dict{T, Vector{Float64}} where T <: Union{PyObject, String};
                                  log::Union{Nothing, Vector{Tuple{Time, String}}}=nothing)
    if !isnothing(log)
        msg = "New CrowdNav control computation is scheduled."
        push!(log, (e_init.t, msg))
    end

    controller.control_update_task = @task begin
        controller.tcalc_actual_tmp, controller.u_value_tmp =
            crowdnav_control_update!(controller, e_init, ado_state_dict);
    end
    schedule(controller.control_update_task);
end

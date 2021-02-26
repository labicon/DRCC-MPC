#///////////////////////////////////////
#// File Name: cost.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/01/15
#// Description: Cost model for Risk Sensitive Stochastic SAC
#///////////////////////////////////////

using LinearAlgebra

# Cost Parameter
mutable struct CostParameter <: Parameter
    #ep_target::Position2D # Goal position for ego robot
    #ep_targets::Trajectory2D
    Cep::Matrix{Float64}  # Quadratic penalty matrix for ego robot position
    Cu::Matrix{Float64}   # Quadratic penalty matrix for ego robot control
    β_pos::Float64        # Relative weight between instant and terminal pos cost
    α_col::Float64           # Magnitude parameter for collision exponential cost.
    β_col::Float64        # Relative weight between instant and terminal col cost
    λ_col::Float64           # Bandwidth parameter for collision exponential cost.
    σ_risk::Float64          # Risk sensitiveness parameter.
end

# Instantaneous Cost
# # Position
function instant_position_cost(e_state::RobotState,
                               target_trajectory::Trajectory2D,
                               param::CostParameter)
    ep_target = get_position(target_trajectory, e_state.t);
    position_error = get_position(e_state) - ep_target;
    position_cost = 0.5*position_error'*param.Cep*position_error;
    return position_cost
end

# # Control
function instant_control_cost(u::Vector{Float64},
                              param::CostParameter)
    @assert length(u) == 2 "Invalid control input dimension!"
    control_cost = 0.5*u'*param.Cu*u;
    return control_cost
end

# # Collision
function instant_collision_cost(e_state::RobotState,
                                apvec::Vector{Float64},
                                param::CostParameter)
    @assert length(apvec) == 2 "Invalid ado state dimension!"
    collision_cost = param.α_col*
                     exp(-1/(2*param.λ_col)*
                     norm(get_position(e_state) - apvec)^2);
    return collision_cost
end

# Terminal Cost
# # Position
terminal_position_cost(e_state::RobotState, target_trajectory::Trajectory2D,
                       param::CostParameter) =
    param.β_pos*instant_position_cost(e_state, target_trajectory, param);

# # Collision
terminal_collision_cost(e_state::RobotState, apvec::Vector{Float64},
                        param::CostParameter) =
    param.β_col*instant_collision_cost(e_state, apvec, param)

# Instantaneous Cost Gradient (with respect to ego robot state)
# # Position
function instant_position_cost_gradient(e_state::RobotState,
                                        target_trajectory::Trajectory2D,
                                        param::CostParameter)
    ep_target = get_position(target_trajectory, e_state.t);
    position_error = get_position(e_state) - ep_target;
    position_cost_gradient = [param.Cep*position_error; zeros(2)];
    return position_cost_gradient
end

# # Collision
function instant_collision_cost_gradient(e_state::RobotState,
                                         apvec::Vector{Float64},
                                         param::CostParameter)
    ep_grad = instant_collision_cost(e_state, apvec, param)/
              param.λ_col*(apvec - get_position(e_state));
    collision_cost_gradient = [ep_grad; zeros(2)];
    return collision_cost_gradient
end

# Terminal Cost Gradient (with respect to ego robot state)
# # Position
terminal_position_cost_gradient(e_state::RobotState,
                                target_trajectory::Trajectory2D,
                                param::CostParameter) =
    param.β_pos*instant_position_cost_gradient(e_state, target_trajectory, param);

# # Collision
terminal_collision_cost_gradient(e_state::RobotState, apvec::Vector{Float64},
                                 param::CostParameter) =
    param.β_col*instant_collision_cost_gradient(e_state, apvec, param);

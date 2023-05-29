using LinearAlgebra

# Cost Parameter
mutable struct DRCCostParameter <: Parameter
    ep_target::Vector{Float64} # Goal position for ego robot
    #ep_targets::Trajectory2D
    Cep::Matrix{Float64}  # Quadratic penalty matrix for ego robot position
    Cu::Matrix{Float64}   # Quadratic penalty matrix for ego robot control
    β_pos::Float64        # Relative weight between instant and terminal pos cost
    α_col::Float64           # Magnitude parameter for collision exponential cost.
    β_col::Float64        # Relative weight between instant and terminal col cost
    λ_col::Float64           # Bandwidth parameter for collision exponential cost.

    human_size::Float64   # Size of human agent
end

# Instantaneous Cost
# # Position
function instant_position_cost(e_state::RobotState,
                                param::DRCCostParameter)
    ep_target = param.ep_target
    position_error = get_position(e_state) - ep_target;
    position_cost = 0.5*position_error'*param.Cep*position_error;
    return position_cost
end

function instant_position_cost(e_state::RobotState,
                                target_trajectory::Trajectory2D,
                                param::DRCCostParameter)
    ep_target = get_position(target_trajectory, e_state.t);
    position_error = get_position(e_state) - ep_target;
    position_cost = 0.5*position_error'*param.Cep*position_error;
    return position_cost
end

# # Control
function instant_control_cost(u::Vector{Float64},
                                param::DRCCostParameter)
    @assert length(u) == 2 "Invalid control input dimension!"
    control_cost = 0.5*u'*param.Cu*u;
    return control_cost
end

# # Collision
function instant_collision_cost(e_state::RobotState,
                                apvec::Vector{Float64},
                                param::DRCCostParameter)
    @assert length(apvec) == 2 "Invalid ado state dimension!"
    collision_cost = param.α_col*
                        exp(-1/(2*param.λ_col)*
                        norm(get_position(e_state) - apvec)^2);
    return collision_cost
end

# # Check Collision
function check_collision(e_state::RobotState,
                            apvec::Vector{Float64},
                            param::DRCCostParameter)
    @assert length(apvec) == 2 "Invalid ado state dimension!"
    if norm(get_position(e_state) - apvec) < param.human_size
        collision = 1
    else
        collision = 0
    end
    return collision
end

# Terminal Cost
# # Position
terminal_position_cost(e_state::RobotState, param::DRCCostParameter) =
    param.β_pos*instant_position_cost(e_state, param);

# # Collision
terminal_collision_cost(e_state::RobotState, apvec::Vector{Float64},
                            param::DRCCostParameter) =
    param.β_col*instant_collision_cost(e_state, apvec, param)
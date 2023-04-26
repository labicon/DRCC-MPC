#///////////////////////////////////////
#// File Name: state_transition.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/01/15
#// Description: State transition model for Risk Sensitive Stochastic SAC
#///////////////////////////////////////

using LinearAlgebra
using Random


# Instantaneous Transition.
function transition(e_state::RobotState,
                    u::Vector{Float64})
    @assert length(u) == 2 "Invalid control input dimension!"
    ev = get_velocity(e_state);
    return [ev; u]
end

function transition_jacobian(ex_vec::Vector{Float64},
                             u::Vector{Float64})
    @assert length(ex_vec) == 4 "Invalid robot state dimension!"
    @assert length(u) == 2 "Invalid control input dimension!"
    A = [0.0 0.0 1.0 0.0;
         0.0 0.0 0.0 1.0;
         0.0 0.0 0.0 0.0;
         0.0 0.0 0.0 0.0];
    return A
end

function transition_jacobian(e_state::RobotState,
                             u::Vector{Float64})
    return transition_jacobian(e_state.x, u)
end

function transition_control_coeff(e_state::RobotState)
    H = [0.0 0.0;
         0.0 0.0;
         1.0 0.0;
         0.0 1.0];
    return H
end

# Forward Euler Integration of Instataneous Transition
function transition(e_state::RobotState,
                    u::Vector{Float64},
                    dt::Real)
    x_new = e_state.x + transition(e_state, u)*dt;
    return RobotState(x_new, e_state.t + Duration(dt))
end

# function transition(e_state::RobotState,
#                     u::Vector{Float64},
#                     dt::Real)
#     dx = [u[1]*dt, u[2]*dt, -e_state.x[1] + u[1], -e_state.x[2] + u[2]]
#     x_new = e_state.x + dx
#     return RobotState(x_new, e_state.t + Duration(dt))
# end
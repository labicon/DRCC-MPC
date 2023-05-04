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
    return [u; 0.0; 0.0]
end

function transition_jacobian(ex_vec::Vector{Float64},
                             u::Vector{Float64})
    @assert length(ex_vec) == 4 "Invalid robot state dimension!"
    @assert length(u) == 2 "Invalid control input dimension!"
    A = [0.0 0.0 0.0 0.0;
         0.0 0.0 0.0 0.0;
         0.0 0.0 0.0 0.0;
         0.0 0.0 0.0 0.0];
    return A
end

function transition_jacobian(e_state::RobotState,
                             u::Vector{Float64})
    return transition_jacobian(e_state.x, u)
end

function transition_control_coeff(e_state::RobotState)
    H = [1.0 0.0;
         0.0 1.0;
         0.0 0.0;
         0.0 0.0];
    return H
end

# Forward Euler Integration of Instataneous Transition
function transition(e_state::RobotState,
                    u::Vector{Float64},
                    dt::Real)
    x_new = e_state.x + transition(e_state, u)*dt;
    x_new[3] = u[1]
    x_new[4] = u[2]
    return RobotState(x_new, e_state.t + Duration(dt))
end
#///////////////////////////////////////
#// File Name: state_transition_test.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/01/15
#// Description: Test code for src/state_transition.jl
#///////////////////////////////////////

using ForwardDiff
using LinearAlgebra
using Random
using RobotOS

@testset "State Transition Test" begin
p1 = [3., 4.];
v1 = [1., 1.];
p2 = [7., 8.];
ap_dict = Dict("Pedestrian/1" => p2);
e1 = RobotState([p1; v1]);
u1 = [2., 2.];
# Continuous Transition
ev1 = transition(e1, u1);
e2 = transition(e1, u1, 0.1);
@test ev1 == [1., 1., 2., 2.]
@test e2.x == [3.1, 4.1, 1.2, 1.2];
@test e2.t == Time(0.1);

function transition_vector(evec::Vector{<:Real},
                           u::Vector{Float64})
    return [evec[3:4]; u]
end
function test_jacobian(e_state::RobotState, u::Vector{Float64})
    return ForwardDiff.jacobian(x -> transition_vector(x, u), e_state.x)
end

@test transition_jacobian(e1.x, u1) == transition_jacobian(e1, u1);
@test transition_jacobian(e1, u1) ≈ test_jacobian(e1, u1);

@test transition(e1, u1) - transition(e1, [0.0, 0.0]) ≈
        transition_control_coeff(e1)*u1;

end

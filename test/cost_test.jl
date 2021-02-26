#///////////////////////////////////////
#// File Name: cost_test.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/01/15
#// Description: Test code for src/cost.jl
#///////////////////////////////////////

using ForwardDiff
using LinearAlgebra
using RobotOS

@testset "Cost Test" begin
target_point = [0.0, 0.0];
wp1 = WayPoint2D(target_point, Time(0.0));
wp2 = WayPoint2D(target_point, Time(1.0));
target_trajectory = Trajectory2D([wp1, wp2]);
Cep = Matrix(1.0I, 2, 2);
Cu = Matrix(1.0I, 2, 2);
β_pos = 0.6;
β_col = 0.4;
α_col = 100.0;
λ_col = 1.0;
σ_risk = 1.0;
param = CostParameter(Cep, Cu, β_pos, α_col, β_col, λ_col, σ_risk);

p1 = [3., 4.];
p2 = [7., 8.];
v1 = [1., 1.];
u1 = [2., 2.];
e1 = RobotState([p1; v1]);

# Instantaneous Cost
# # Position
cip = instant_position_cost(e1, target_trajectory, param);
@test cip ≈ 1/2*(p1 - target_point)'*Cep*(p1 - target_point);
# # Control
ciu = instant_control_cost(u1, param);
@test ciu ≈ 1/2*u1'*Cu*u1;
# # Collision
cic = instant_collision_cost(e1, p2, param);
@test cic ≈ α_col*exp(-1/(2*λ_col)*norm(p1 - p2)^2);

# Terminal Cost
# # Position
ctp = terminal_position_cost(e1, target_trajectory, param);
@test ctp ≈ param.β_pos*cip;
# # Collision
ctc = terminal_collision_cost(e1, p2, param);
@test ctc ≈ param.β_col*cic;

# Instantaneous Cost Gradient
# # Position
function instant_position_cost_vector(evec::Vector{<:Real},
                                      t::Time,
                                      target_trajectory::Trajectory2D,
                                      param::CostParameter)
    ep_target = get_position(target_trajectory, t)
    ep = evec[1:2];
    return 1/2*(ep - ep_target)'*param.Cep*(ep - ep_target);
end
function test_instant_position_cost_gradient(e_state::RobotState,
                                             target_trajectory::Trajectory2D,
                                             param::CostParameter)
    return ForwardDiff.gradient(x -> instant_position_cost_vector(x, e1.t,
                                target_trajectory, param), e1.x)
end
gip = instant_position_cost_gradient(e1, target_trajectory, param);
@test gip ≈ test_instant_position_cost_gradient(e1, target_trajectory, param);
# # Collision
function instant_collision_cost_vector(evec::Vector{<:Real},
                                       apvec::Vector{<:Real},
                                       param::CostParameter)
    ep = evec[1:2]
    return param.α_col*exp(-1/(2*param.λ_col)*norm(ep - apvec)^2);
end
function test_instant_collision_cost_gradient(e_state::RobotState,
                                              apvec::Vector{<:Real},
                                              param::CostParameter)
    return ForwardDiff.gradient(x -> instant_collision_cost_vector(x,
                                apvec, param), e_state.x)
end
gic = instant_collision_cost_gradient(e1, p2, param);
@test gic ≈ test_instant_collision_cost_gradient(e1, p2, param);

# Terminal Cost Gradient
# # Position
function test_terminal_position_cost_gradient(e_state::RobotState,
                                             target_trajectory::Trajectory2D,
                                             param::CostParameter)
    return ForwardDiff.gradient(x -> param.β_pos*instant_position_cost_vector(x, e1.t,
                                                 target_trajectory, param), e1.x)
end
gtp = terminal_position_cost_gradient(e1, target_trajectory, param);
@test gtp ≈ test_terminal_position_cost_gradient(e1, target_trajectory, param);
# # Collision
function test_terminal_collision_cost_gradient(e_state::RobotState,
                                               apvec::Vector{<:Real},
                                               param::CostParameter)
    return ForwardDiff.gradient(x -> param.β_col*instant_collision_cost_vector(x,
                                                 apvec, param), e_state.x)
end
gtc = terminal_collision_cost_gradient(e1, p2, param);
@test gtc ≈ test_terminal_collision_cost_gradient(e1, p2, param);

end

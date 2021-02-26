#///////////////////////////////////////
#// File Name: type_definition_test.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/01/15
#// Description: Test code for src/type_definition.jl
#///////////////////////////////////////

using Distributions
using LinearAlgebra
using RobotOS

@testset "Type Definition Test" begin

# WayPoint & Trajectory
p1 = [3., 4.];
p2 = [2., 1.];
wp1 = WayPoint2D(p1, Time(0.0));
wp2 = WayPoint2D(p2, Time(1.0));
traj = Trajectory2D([wp1, wp2]);
@test get_position(traj, Time(0.0)) == p1;
@test get_position(traj, Time(0.1)) ≈ [2.9, 3.7];
@test get_position(traj, Time(1.0)) == p2;

# Robot State
e1 = RobotState([p1; 0.; 0.]);
@test e1.x == [3., 4., 0., 0.];
@test get_position(e1) == [3., 4.];
@test get_velocity(e1) == [0., 0.];
@test e1 == RobotState([p1; 0.; 0;], Time(0.0));
@test e1 ≈ e1;

# World State
ap_dict = Dict("Pedestrian/1" => p2);
w = WorldState(e1, ap_dict);
@test w.t == Time(0.0);
@test isapprox(w, w);

end

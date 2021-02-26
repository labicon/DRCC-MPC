#///////////////////////////////////////
#// File Name: type_definition.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/01/15
#// Description: Type definitions for Risk Sensitive Stochastic SAC
#///////////////////////////////////////

import Base: ==, isequal, isapprox
import RobotOS: Time

using DataStructures
using LinearAlgebra
using RobotOS


abstract type Parameter end;

abstract type State end;

# WayPoint
struct WayPoint2D
    pos::Vector{Float64}
    t::Time
    function WayPoint2D(pos::Vector{Float64}, t::Time)
        @assert length(pos) == 2 "Invalid waypoint dimension!"
        return new(pos, t)
    end
end

# Trajectory
const Trajectory2D = OrderedDict{Time, Vector{Float64}};

function Trajectory2D(points::Vector{WayPoint2D})
    trajectory = Trajectory2D();
    for waypoint in points
        trajectory[waypoint.t] = waypoint.pos;
    end
    sort!(trajectory)
    return trajectory
end

function get_position(trajectory::Trajectory2D, t::Time)
    if haskey(trajectory, t)
        return trajectory[t]
    else
        traj_array = collect(trajectory);
        index2 = findfirst(x -> x[1] > t, traj_array);
        @assert !(index2 == 1 || index2 == nothing) "Trajectory Interpolation failed for t = $(to_sec(t)) [s]!"
        t2 = traj_array[index2][1];
        index1 = index2 - 1;
        t1 = traj_array[index1][1];
        # linear interpolation
        pos = traj_array[index1][2] + to_nsec(t - t1)/to_nsec(t2 - t1)*
              (traj_array[index2][2] - traj_array[index1][2])
        return pos
    end
end


# Ego Robot State
struct RobotState <: State
    x::Vector{Float64} # [pos_x, pos_y, vel_x, vel_y]
    t::Time
    function RobotState(x::Vector{Float64}, t::Time)
        @assert length(x) == 4 "Invalid ego state dimension!"
        return new(x, t)
    end
end

RobotState(x::Vector{Float64}) = RobotState(x, Time());

function isequal(e1::RobotState, e2::RobotState)
    return e1.x == e2.x && e1.t == e2.t
end

==(e1::RobotState, e2::RobotState) = isequal(e1, e2)

function isapprox(e1::RobotState, e2::RobotState; kwargs...)
    return isapprox(e1.x, e2.x; kwargs...) && e1.t == e2.t
end

get_position(e::RobotState) = e.x[1:2];
get_velocity(e::RobotState) = e.x[3:4];

# Joint State
mutable struct WorldState <: State
    e_state::RobotState
    ap_dict::Dict{String, Vector{Float64}}
    t::Time
    t_last_m::Union{Time, Nothing}
    function WorldState(e_state::RobotState,
                        ap_dict::Dict{String, Vector{Float64}},
                        t::Time, t_last_m::Union{Time, Nothing})
        @assert all(length.(values(ap_dict)) .== 2) "Invalid ado state dimension!"
        @assert e_state.t == t "Inconsistent time stamps!";
        if t_last_m != nothing
            @assert t_last_m <= t "Last measurement time cannot be in future!"
        end
        return new(e_state, ap_dict, t, t_last_m)
    end
end

function WorldState(e_state::RobotState,
                    ap_dict::Dict{String, Vector{Float64}},
                    t_last_m=nothing)
    return WorldState(e_state, ap_dict, e_state.t, t_last_m)
end

function WorldState(x::Vector{Float64},
                    ap_dict::Dict{String, Vector{Float64}},
                    t::Time=Time(), t_last_m=nothing)
    return WorldState(RobotState(x, t), ap_dict, t, t_last_m)
end
function isequal(w1::WorldState, w2::WorldState)
    return begin w1.e_state == w2.e_state && w1.ap_dict == w2.ap_dict &&
                 w1.t == w2.t && w1.t_last_m == w2.t_last_m
           end
end

==(w1::WorldState, w2::WorldState) = isequal(w1, w2);

function isapprox(w1::WorldState, w2::WorldState; kwargs...)
    e_approx = isapprox(w1.e_state, w2.e_state; kwargs...);
    ap_approx = keys(w1.ap_dict) == keys(w2.ap_dict) &&
                all([isapprox(w1.ap_dict[key], w2.ap_dict[key]; kwargs...)
                     for key in keys(w1.ap_dict)]);
    return e_approx && ap_approx && w1.t == w2.t && w1.t_last_m == w2.t_last_m
end

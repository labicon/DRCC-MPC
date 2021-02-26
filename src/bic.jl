#///////////////////////////////////////
#// File Name: bic.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/02/25
#// Description: Buffered Input Cells and Related Functions for Benchmarking
#///////////////////////////////////////

import Convex: Variable, norm, minimize, dot, solve!
using GeometricalPredicates
using LazySets
using LinearAlgebra
using Printf
using Polyhedra
using Random
using RobotOS
using SCS
using VoronoiCells


# Buffered Voronoi Cells
# # helper function for sorting Voronoi vertices couterclock-wise
function vertex_direction(generator::IndexablePoint2D, vertex::Point2D)
    x = getx(vertex) - getx(generator)
    y = gety(vertex) - gety(generator)
    return atan(y, x)
end

# # sort Voronoi vertices counterclock-wise
function sort_vertices(generator::IndexablePoint2D, vertices::Vector{Point2D})
    return sort(vertices, by=x->vertex_direction(generator, x))
end

# # scale and shift points (to fit a certain rectangle, expecially [1,2]x[1,2])
function scale_and_shift(x::Real; scale_factor::Real, subtract::Real,
                         offset::Real)
    return x = scale_factor*(x - subtract) + offset
end

# # find extrema of coordinates of all points in the scene
function find_extrema(input_positions::Vector{Vector{Float64}})
    xmin = minimum([p[1] for p in input_positions]);
    xmax = maximum([p[1] for p in input_positions]);
    offset_x = 5. + (xmax - xmin);
    xmin -= offset_x; # additional offset
    xmax += offset_x; # additinal offset
    ymin = minimum([p[2] for p in input_positions]);
    ymax = maximum([p[2] for p in input_positions]);
    offset_y = 5. + (ymax - ymin);
    ymin -= offset_y; # additional offset
    ymax += offset_y; # additional offset
    return xmin, xmax, ymin, ymax
end

# # compute buffered voronoi cell from standard voronoi cell
function retract_cell(generator::IndexablePoint2D,
                      vertices::Vector{Vector{Float64}},
                      radius::Real)
    vpoly = VPolygon(vertices)
    ball = Ball2(zeros(2), radius);
    hpoly = minkowski_difference(vpoly, ball)
    point_vec_new = vertices_list(tovrep(hpoly))
    points = Point2D[]
    for point in point_vec_new
        push!(points, Point2D(point[1], point[2]))
    end
    if isempty(points) # bvc is an empty set. Returning generator.
        p = Point2D(getx(generator), gety(generator))
        return Polygon2D(p)
    else
        return Polygon2D(sort_vertices(generator, points)...)
    end
end

# # extract all agent positions from WorldState
function get_input_positions(w_init::WorldState)
    input_positions = Vector{Vector{Float64}}()
    # ego is always the first element
    push!(input_positions, get_position(w_init.e_state));
    for ap in values(w_init.ap_dict)
        push!(input_positions, ap)
    end
    return input_positions
end

# # main function to compute buffered voronoi cells (idx 1 is for ego robot)
function compute_buffered_voronoi_cells(w_init::WorldState,
                                        radius::Real)
    # compute input positions
    input_positions = get_input_positions(w_init);
    xmin, xmax, ymin, ymax = find_extrema(input_positions)

    # scale and shift
    scaled_positions = [[scale_and_shift(p[1], scale_factor=1. /(xmax - xmin),
                                               subtract=xmin, offset=1.),
                         scale_and_shift(p[2], scale_factor=1. /(ymax - ymin),
                                               subtract=ymin, offset=1.)]
                         for p in input_positions]
    generators = [IndexablePoint2D(p[1], p[2], ii) for (ii, p) in
                  enumerate(scaled_positions)];

    # compute voronoi cells (vertices)
    scaled_voronoi_cells = voronoicells(generators)

    # unshift and unscale
    vertices_vec = Vector{Vector{Vector{Float64}}}(undef, length(generators));
    for ii = 1:length(generators)
        vertices_vec[ii] = [[scale_and_shift(getx(p), scale_factor=(xmax - xmin),
                                                      subtract=1., offset=xmin),
                             scale_and_shift(gety(p), scale_factor=(ymax - ymin),
                                                      subtract=1., offset=ymin)]
                             for p in sort_vertices(generators[ii],
                                                    scaled_voronoi_cells[ii])];
    end

    # compute buffered voronoi cells (polygons)
    buffered_voronoi_cells = Dict{Int64, Polygon2D}()
    for ii = 1:length(generators)
        points = [Point2D(p[1], p[2]) for p in vertices_vec[ii]];
        #buffered_voronoi_cells[ii] = Polygon2D(points...)
        generator = IndexablePoint2D(input_positions[ii][1],
                                     input_positions[ii][2], ii)
        buffered_voronoi_cells[ii] = retract_cell(generator, vertices_vec[ii],
                                                  radius)
    end
    return buffered_voronoi_cells
end

# # robot's dynamics (exact double-integrator, not Euler approximation)
C() = [1. 0. 0. 0.;
       0. 1. 0. 0.]; # state [px, py, vx, vy] to position [px, py]
F(dt) = [1. 0. dt 0.;
         0. 1. 0. dt;
         0. 0. 1. 0.;
         0. 0. 0. 1.]; # dynamics matrix "A"

G(dt) = [0.5*dt^2  0.;
         0.        0.5*dt^2;
         dt        0.;
         0.        dt]; # dynamics matrix "B"

# # main function to compute buffered input cell
function compute_buffered_input_cell(bvc::Polygon2D, # buffered voronoi cell
                                     e_state::RobotState,
                                     dt::Float64)
    x = e_state.x;
    Cm = C();
    Fm = F(dt);
    Gm = G(dt);
    J = Cm*Gm;
    vertices_bvc = [[getx(p), gety(p)] for p in getpoints(bvc)];
    vertices_bic = similar(vertices_bvc);
    for ii = 1:length(vertices_bvc)
        vertices_bic[ii] = J\(vertices_bvc[ii] - Cm*Fm*x);
    end
    return Polygon2D([Point2D(p[1], p[2]) for p in vertices_bic]...)
end

# # project a given 2D point onto buffered input cell (represented as Polygon2D)
function project(point::Vector{Float64}, bic::Polygon2D)
    # QP solver to project target control input to bic
    vertices = [[getx(p), gety(p)] for p in getpoints(bic)];
    vpoly = VPolygon(vertices);
    hpoly = tohrep(vpoly);

    x = Variable(2);
    objective = norm(point - x, 2) # L2 norm between point and p.
    problem = minimize(objective)
    for c in hpoly.constraints
        problem.constraints += dot(c.a, x) <= c.b
    end
    solve!(problem, () -> SCS.Optimizer(verbose=false))
    return Float64[x.value[1], x.value[2]]
end


# LQ Tracker
function lq_tracking(w_init::WorldState,
                     target_trajectory::Trajectory2D,
                     sim_param::SimulationParameter)
    # Reference: Optimal Control by Frank L. Lewis, Table 4.4-1

    # retrieve waypoints from target_trajectory
    prediction_horizon = sim_param.dto*sim_param.prediction_steps;
    ts_times = get_measurement_schedule(w_init, prediction_horizon, sim_param);
    pushfirst!(ts_times, w_init.t); # [t_1, t_2, ..., t_N]
    target_state_array = [[get_position(target_trajectory, t)..., 0.0, 0.0]
                           for t in ts_times]; # [r_1, r_2, ..., r_N, r_N+1]
                                               # final time idx == N+1

    # compute gains
    Gm = G(sim_param.dto);
    Fm = F(sim_param.dto);

    S = Vector{Matrix{Float64}}(undef, sim_param.prediction_steps + 1) # [S_1, S_2, ..., S_N, S_N+1]
    K = Vector{Matrix{Float64}}(undef, sim_param.prediction_steps)     # [K_1, K_2, ..., K_N]
    v = Vector{Vector{Float64}}(undef, sim_param.prediction_steps + 1) # [v_1, v_2, ..., v_N, v_N+1]
    H = Vector{Matrix{Float64}}(undef, sim_param.prediction_steps)     # [H_1, H_2, ..., H_N]

    Q = zeros(4, 4); # state x is [px, py, vx, vy];
    Q[1:2, 1:2] = sim_param.cost_param.Cep;
    R = sim_param.cost_param.Cu;
    S[end] = Q;
    v[end] = Q*target_state_array[end];

    for ii = Iterators.reverse(1:sim_param.prediction_steps)
        K[ii] = (Gm'*S[ii + 1]*Gm + R)\Gm'*S[ii + 1]*Fm;
        S[ii] = Fm'*S[ii + 1]*(Fm - Gm*K[ii]) + Q;
        v[ii] = (Fm - Gm*K[ii])'*v[ii + 1] + Q*target_state_array[ii];
        H[ii] = (Gm'*S[ii + 1]*Gm + R)\Gm'
    end

    # compute optimal control
    u_array = Vector{Vector{Float64}}(undef, sim_param.prediction_steps);    # [u_1, u_2, ..., u_N]
    x_array = Vector{Vector{Float64}}(undef, sim_param.prediction_steps + 1);# [x_1, x_2, ..., x_N, x_N+1]
    x_array[1] = w_init.e_state.x;
    for ii = 1:sim_param.prediction_steps
        u_array[ii] = -K[ii]*x_array[ii] + H[ii]*v[ii + 1];
        x_array[ii + 1] = Fm*x_array[ii] + Gm*u_array[ii]
    end

    return u_array, x_array
end


# BIC Controller
# # bic control parameter & controller
struct BICControlParameter <: Parameter
    eamax::Float64 # Maximum absolute value of accleration
    tcalc::Float64 # Allocated control computation time
    dtr::Float64   # Replanning time interval
    min_dist::Float64 # Minimum allowable distance between ego and ados

    function BICControlParameter(eamax::Float64, tcalc::Float64, dtr::Float64,
                                 min_dist::Float64)
        @assert tcalc <= dtr
        return new(eamax, tcalc, dtr, min_dist)
    end
end

mutable struct BICController
    sim_param::SimulationParameter
    cnt_param::BICControlParameter

    tcalc_actual::Union{Nothing, Float64}
    u_value::Union{Nothing, Vector{Float64}}

    control_update_task::Union{Nothing, Task}

    tcalc_actual_tmp::Union{Nothing, Float64}
    u_value_tmp::Union{Nothing, Vector{Float64}}
end

function BICController(sim_param::SimulationParameter,
                       cnt_param::BICControlParameter)
    return BICController(sim_param, cnt_param, nothing, nothing, nothing,
                         nothing, nothing)
end

# # bic control update
@inline function bic_control_update(w_init::WorldState,
                                    target_trajectory::Trajectory2D,
                                    sim_param::SimulationParameter,
                                    cnt_param::BICControlParameter)
    # LQ Tracker
    tcalc_actual = @elapsed u_array, ~ =
        lq_tracking(w_init, target_trajectory, sim_param);

    # compute buffered input cell
    tcalc_actual +=
        @elapsed bvcs = compute_buffered_voronoi_cells(w_init,
                                                       cnt_param.min_dist/2.);

    tcalc_actual +=
        @elapsed bic_ego = compute_buffered_input_cell(bvcs[1], w_init.e_state,
                                                       sim_param.dto);

    # project u_array[1] to bic_ego
    tcalc_actual +=
        @elapsed u_projected = project(u_array[1], bic_ego);

    # saturate u_projected
    if norm(u_projected) > cnt_param.eamax
        u_projected = u_projected./norm(u_projected).*cnt_param.eamax;
    end

    if tcalc_actual >= cnt_param.tcalc
        # tcalc_actual has exceeded allowable computation time
        time = @sprintf "Time %.2f" round(to_sec(w_init.t), digits=5)
        @warn "$(time) [sec]: BIC computation took $(round(tcalc_actual, digits=3)) [sec], which exceeds the maximum computation time allowed."
    end

    return tcalc_actual, u_projected
end

# # main control function for bic
function control!(controller::BICController,
                  current_time::Time;
                  log::Union{Nothing, Vector{Tuple{Time, String}}}=nothing)

    if !isnothing(controller.control_update_task) &&
        istaskdone(controller.control_update_task)
        if !isnothing(log)
            msg = "New BIC control is available to the controller."
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

# # control update schedule function for bic
function schedule_control_update!(controller::BICController,
                                  w_init::WorldState,
                                  target_trajectory::Trajectory2D;
                                  log::Union{Nothing, Vector{Tuple{Time, String}}}=nothing)
    if !isnothing(log)
        msg = "New BIC control computation is scheduled."
        push!(log, (w_init.t, msg))
    end

    controller.control_update_task = @task begin
        controller.tcalc_actual_tmp, controller.u_value_tmp =
            bic_control_update(w_init, target_trajectory,
                               controller.sim_param,
                               controller.cnt_param);
    end
    schedule(controller.control_update_task);
end

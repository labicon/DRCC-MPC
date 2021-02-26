#///////////////////////////////////////
#// File Name: bic_test.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/02/25
#// Description: Test code for src/bic.jl
#///////////////////////////////////////

using Distributions
using GeometricalPredicates
using LinearAlgebra
using Random
using RobotOS
using VoronoiCells

@testset "Buffered Input Cell Test" begin
# # vertex_direction test
generator = IndexablePoint2D(0.0, 0.0, 1);
vertex = Point2D(1.0, 1.0);
@test vertex_direction(generator, vertex) ≈ pi/4;
# # sort_vertices test
vertices = [Point2D(1.0, 1.0),
            Point2D(1.0, -1.0),
            Point2D(-1.0, -1.0),
            Point2D(-1.0, 1.0)];
vertices_sorted = sort_vertices(generator, vertices);
@test vertices_sorted == [Point2D(-1.0, -1.0),
                          Point2D(1.0, -1.0),
                          Point2D(1.0, 1.0),
                          Point2D(-1.0, 1.0)];
# # scale_and_shift test
x_test = 10.0;
@test scale_and_shift(x_test, scale_factor=0.1;
                      subtract=2.0, offset=1.0) ≈ 1.8;
# # find_extrema test
input_positions_test = [[-1.0, -1.0],
                        [1.0,  -1.0],
                        [1.0,   1.0],
                        [-1.0,  1.0]];
xmin, xmax, ymin, ymax = find_extrema(input_positions_test);
@test (xmin, xmax) == (-8.0, 8.0);
@test (ymin, ymax) == (-8.0, 8.0);
# # retract_cell test
vertices = input_positions_test
cell = retract_cell(generator, vertices, 0.2)
points = getpoints(cell);
@test points == [Point2D(-0.8,-0.8),
                 Point2D(0.8, -0.8),
                 Point2D(0.8,  0.8),
                 Point2D(-0.8, 0.8)];
# # get_input_positions test
e_init = RobotState([0.0, 0.0, 0.0, 0.0]);
ap_dict = Dict{String, Vector{Float64}}()
ap_dict["Pedestrian/1"] = [1.0,   1.0];
ap_dict["Pedestrian/2"] = [1.0,  -1.0];
ap_dict["Pedestrian/3"] = [-1.0,  1.0];
ap_dict["Pedestrian/4"] = [-1.0, -1.0];
w_init = WorldState(e_init, ap_dict);
input_positions = get_input_positions(w_init);
@test input_positions == [[0.0,   0.0],
                          [1.0,   1.0],
                          [1.0,  -1.0],
                          [-1.0,  1.0],
                          [-1.0, -1.0]];
# # compute_buffered_voronoi_cells test
bvc = compute_buffered_voronoi_cells(w_init, 0.2)
points = getpoints(bvc[1]);
@test all(isapprox.((getx(points[1]), gety(points[1])), (0.0, -(1 - 0.2*sqrt(2))), atol=1e-8));
@test all(isapprox.((getx(points[2]), gety(points[2])), (1 - 0.2*sqrt(2), 0.0), atol=1e-8));
@test all(isapprox.((getx(points[3]), gety(points[3])), (0.0, 1 - 0.2*sqrt(2)), atol=1e-8));
@test all(isapprox.((getx(points[4]), gety(points[4])), (-(1 - 0.2*sqrt(2)), 0.0), atol=1e-8));
# # compute_buffered_voronoi_cells test (empty cell case)
bvc2 = compute_buffered_voronoi_cells(w_init, 1.0)
points2 = getpoints(bvc2[1]);
@test length(points2) == 1;
@test all((getx(points2[1]), gety(points2[1])) .≈ (0.0, 0.0))
# # compute_buffered_input_cell test
bic = compute_buffered_input_cell(bvc[1], w_init.e_state, 0.4);
points = getpoints(bic)
@test all(isapprox.((getx(points[1]), gety(points[1])), (0.0, -2/(0.4^2)*(1 - 0.2*sqrt(2))), atol=1e-8));
@test all(isapprox.((getx(points[2]), gety(points[2])), (2/(0.4^2)*(1 - 0.2*sqrt(2)), 0.0), atol=1e-8));
@test all(isapprox.((getx(points[3]), gety(points[3])), (0.0, 2/(0.4^2)*(1 - 0.2*sqrt(2))), atol=1e-8));
@test all(isapprox.((getx(points[4]), gety(points[4])), (-2/(0.4^2)*(1 - 0.2*sqrt(2)), 0.0), atol=1e-8));

projected = project([100., 0.], bic);
@test all(isapprox.(projected, [2/(0.4^2)*(1 - 0.2*sqrt(2)), 0.], atol=1e-8));
end

@testset "LQ Tracking Test" begin
# # lq_tracking test
target_trajectory = Trajectory2D()
target_trajectory[Time(0.0)] = [0.0, 0.0];
target_trajectory[Time(10.0)] = [10.0, 0.0];

e_init = RobotState([0.0, 0.0, 1.0, 0.0]);
ap_dict = Dict{String, Vector{Float64}}()
ap_dict["Pedestrian/1"] = [1.0,   1.0];
w_init = WorldState(e_init, ap_dict);

Cep = Matrix(0.5I, 2, 2);
Cu = Matrix(0.2I, 2, 2);
β_pos = 0.6;
β_col = 0.4;
α_col = 100.0;
λ_col = 0.2;
σ_risk = 0.0;
cost_param = CostParameter(Cep, Cu, β_pos, α_col, β_col, λ_col, σ_risk);

dtc = 0.02;
dto = 0.4;
prediction_steps = 12;
num_samples = 30;
sim_param = SimulationParameter(dtc, dto, prediction_steps, num_samples, cost_param)

u_array, x_array = lq_tracking(w_init, target_trajectory, sim_param);
for ii = 1:length(u_array)
    @test all(isapprox.(u_array[ii], [0.0, 0.0], atol=1e-8))
    @test all(isapprox.(x_array[ii], [0.4*(ii - 1),  0.0, 1.0, 0.0], atol=1e-8))
end
end

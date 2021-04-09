#///////////////////////////////////////
#// File Name: crowd_nav_controller_test.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/04/08
#// Description: Test code for src/crowd_nav_controller.jl
#///////////////////////////////////////

using LinearAlgebra
using PyCall

@testset "Crowd Nav Controller Test" begin
    Cep = Matrix(0.5I, 2, 2);                                                           # quadratic position cost matrix
    Cu = Matrix(0.2I, 2, 2);                                                            # quadratic control cost matrix
    β_pos = 0.1;                                                                        # relative weight between instant and terminal pos cost
    α_col = 100.0;                                                                      # scale parameter for exponential collision cost
    β_col = 0.1;                                                                        # relative weight between instant and terminal pos cost
    λ_col = 0.2;                                                                        # bandwidth parameter for exponential collision cost
    σ_risk = 0.0;                                                                       # risk-sensitiveness parameter

    cost_param = CostParameter(Cep, Cu, β_pos, α_col, β_col, λ_col, σ_risk);
    sim_param = SimulationParameter(0.01, 0.4, 12, 10, cost_param);

    tcalc = 0.1;
    dtr = 0.4;
    model_dir = normpath(joinpath(@__DIR__,
                                  "../CrowdNav/crowd_nav/data/output_om_sarl_radius_0.4"))
    cnt_param = CrowdNavControlParameter(model_dir,
                                         "env.config", "policy.config", "sarl",
                                         [0.0, 4.0], tcalc, dtr);

    crowd_nav_controller = CrowdNavController(sim_param, cnt_param);
    @test !crowd_nav_controller.rl_robot.visible
    @test crowd_nav_controller.rl_robot.policy.time_step == 0.4;
    @test crowd_nav_controller.rl_robot.time_step == 0.4;
    @test crowd_nav_controller.rl_robot.gx == 0.0;
    @test crowd_nav_controller.rl_robot.gy == 4.0;

    e_init = RobotState([0.0, -4.0, 0.0, 0.0]);
    u_vel = get_action!(crowd_nav_controller, e_init,
                        Dict{String, Vector{Float64}}());
    @test u_vel ≈ [0.0, 8.0]./norm([0.0, 8.0])*crowd_nav_controller.rl_robot.v_pref

    e_init = RobotState([0.0, 0.0, 0.0, 0.0]);
    ado_state_dict = Dict{String, Vector{Float64}}();
    ado_state_dict["PEDESTRIAN/1"] = [5.0, 4.0, -1.0, -3.0];
    ado_state_dict["PEDESTRIAN/2"] = [-3.0, 0.0, 1.0, 1.0];

    u_vel = get_action!(crowd_nav_controller, e_init, ado_state_dict)

    ~, u_acc = crowdnav_control_update!(crowd_nav_controller, e_init, ado_state_dict)
    #@test all(u_vel.*0.4 .≈ get_velocity(e_init).*0.4 .+ 0.5*u_acc.*(0.4)^2)
    @test all(u_vel .≈ get_velocity(e_init) .+ u_acc.*0.4)
end

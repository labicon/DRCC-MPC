#///////////////////////////////////////
#// File Name: runtests.jl
#// Author: Haruki Nishimura (hnishimura@stanford.edu)
#// Date Created: 2021/01/15
#// Description: Test script for RiskSensitiveSAC package
#///////////////////////////////////////

using Test
using RiskSensitiveSAC
using PyCall

py"""
import torch
"""

@testset "RiskSensitiveSAC Unit Tests" begin
@info "Executing Type Definition Test"
include("type_definition_test.jl");
@info "Executing State Transition Test"
include("state_transition_test.jl");
@info "Executing Cost Test"
include("cost_test.jl")
@info "Executing Cost CUDA Test"
include("cost_gpu_test.jl")
@info "Executing Scene Loader Test"
include("scene_loader_test.jl")
@info "Executing Predictor Test"
include("predictor_test.jl")
@info "Executing Forward-Backward Simulation Test"
include("forward_backward_simulation_test.jl")
@info "Executing RiskSensitiveSAC Controller Test"
include("rs_sac_controller_test.jl")
@info "Executing BIC Test"
include("bic_test.jl")
@info "Executing CrowdNav Test"
include("crowd_nav_controller_test.jl")
@info "Executing Utils Test"
include("utils_test.jl")
end

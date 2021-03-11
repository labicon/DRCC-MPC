using LinearAlgebra, Distributions
ado_vel_dict = Dict("PEDESTRIAN/1" => MvNormal([0.0, 1.0], Diagonal([0.04, 0.60])));# ado velocity distributions

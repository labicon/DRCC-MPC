dtc = 0.02;                                                                         # Euler integration time interval [s]
dtr = 0.1;                                                                          # replanning time interval [s]
dtexec = [0.0, 0.001, 0.002, 0.004, 0.008, 0.016, 0.02, 0.04, 0.08];                # control insertion duration [s]
tcalc = 0.1;                                                                        # pre-allocated control computation time [s] (< dtr)
u_norm_max = 2.0;                                                                   # maximum control norm [m/s^2]
u_nominal_base = [0.0, 0.0];                                                        # nominal control base.
#β = 1.6;                                                                            # penalty for waiting time
improvement_threshold = 0.0;
constraint_time = nothing
u_nominal_cand = append!([u_nominal_base],
                         [round.([a*cos(deg2rad(θ)), a*sin(deg2rad(θ))], digits=5)
                          for a = [1., 2.] for θ = 0.:45.:(360. - 45.)])            # nominal control candidate value [ax, ay] [m/s^2]
nominal_search_depth = 1;
#improvement_threshold = 0.1;
#constraint_time = tcalc + dtr;
dtc = 0.4;                                                                         # Euler integration time interval [s]
dtr = 0.4;                                                                          # replanning time interval [s]
tcalc = 0.1;                                                                        # pre-allocated control computation time [s] (< dtr)
u_norm_max = 1.7;                                                                   # maximum control norm [m/s^2]
constraint_time = nothing

horizon = 10;

human_size = 0.5;

cem_init_mean = [0.0, 0.0]
cem_init_cov = Matrix(1.0I, 2, 2) * u_norm_max^2
cem_init_num_samples = 400
cem_init_num_elites = 40
cem_init_alpha = 0.2
cem_init_iterations = 10

epsilon = 0.05
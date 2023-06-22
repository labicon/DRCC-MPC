dtc = 0.1;                                                                         # Euler integration time interval [s]
dtr = 0.1;                                                                          # replanning time interval [s]
tcalc = 0.1;                                                                        # pre-allocated control computation time [s] (< dtr)
u_norm_max = 2.0;                                                                   # maximum control norm [m/s^2]
constraint_time = nothing

horizon = 40;
discount = 0.99;

human_size = 0.4;

cem_init_mean = [0.0, 0.0]
cem_init_cov = Matrix(1.0I, 2, 2) * u_norm_max^2
cem_init_num_samples = 400
cem_init_num_elites = 40
cem_init_alpha = 0.8
cem_init_iterations = 5

epsilon = 0.05
prediction_device = "cpu";
u_nominal_cand = append!([u_nominal_base],
                         [round.([a*cos(deg2rad(θ)), a*sin(deg2rad(θ))], digits=5)
                          for a = [3.] for θ = 0.:45.:(360. - 45.)])            # nominal control candidate value [ax, ay] [m/s^2]
nominal_search_depth = 4;
prediction_steps = 4;
dtr = 0.4;
tcalc = 0.4;
dtexec = [0.0];
constraint_time = nothing;

dtr = 0.4;                                                                          # replanning time interval [s]
tcalc = 0.2;                                                                        # pre-allocated control computation time [s] (< dtr)

model_dir = normpath(joinpath(@__DIR__, "../CrowdNav/crowd_nav/data/output_om_sarl_radius_0.4")) # directory of the trained policy
env_config = "env.config"                                                           # environment config file name
policy_config = "policy.config"                                                     # policy config file name
policy_name = "sarl"                                                                # policy name

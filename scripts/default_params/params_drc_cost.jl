using LinearAlgebra

Cep = Matrix(1.0I, 2, 2);                                                           # quadratic position cost matrix
Cu = Matrix(0.1I, 2, 2);                                                            # quadratic control cost matrix
β_pos = 0.1;                                                                        # relative weight between instant and terminal pos cost
α_col = 100.0;                                                                      # scale parameter for exponential collision cost
β_col = 0.1;                                                                        # relative weight between instant and terminal pos cost
λ_col = 0.2;                                                                        # bandwidth parameter for exponential collision cost
σ_risk = 0.0;                                                                       # risk-sensitiveness parameter

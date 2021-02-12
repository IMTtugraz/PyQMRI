The config file (\*.ini)
========================
A default config file will be generated if no path to a config file is passed as an argument or if no default.ini file is present in the current working directory. After the initial generation the values can be altered to influence regularization or the number of iterations. Seperate values for TV and TGV regularization can be used. 

- max_iters: Maximum primal-dual (PD) iterations
- start_iters: PD iterations in the first Gauss-Newton step
- max_gn_it: Maximum number of Gauss Newton iterations
- lambd: Data weighting
- gamma: TGV weighting
- delta: L2-step-penalty weighting (inversely weighted)
- omega: optional H1 regularization (should be set to 0 if no H1 is used)
- display_iterations: Flag for displaying grafical output
- gamma_min: Minimum TGV weighting
- delta_max: Maximum L2-step-penalty weighting
- omega_min: Minimum H1 weighting (should be set to 0 if no H1 is used)
- tol: relative convergence toleranze for PD and Gauss-Newton iterations
- stag: optional stagnation detection between successive PD steps
- delta_inc: Increase factor for delta after each GN step
- gamma_dec: Decrease factor for gamma after each GN step
- omega_dec: Decrease factor for omega after each GN step
- beta: The initial ratio between primal and dual step size of the PD algorithm. Will be adapted during the linesearch.

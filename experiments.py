from models import *
from experiments_utils import *

print("Doing MVN experiment")
# MVN - experiment
dim = 20
mvn = mvn(dim=dim, off_diag=0.99)
U, grad_U = mvn.params()

### Test cases for all the methods
start_position = np.repeat(3., dim)
iter = 1000

### Extra pars definition
extra_pars = dict({
    'L_start': 25,
    'L_noise': 0.2,
    'epsilon_noise': 0.2,
    'epsilon_start': 0.25,
    'epsilon_bar': 1,
    'gamma': 0.05,
    't0': 10,
    'kappa': 0.75,
    'adapt_epsilon': 100,
    'threshold': 180,
    'adapt_L': 100,
    'quantile_ub': 0.95,
    'quantile_lb': 0.00,
    'delta_max': 1000,
    'refreshment_prob': 0.4,
    'adapt_opt_epsilon': True,
    'samp_random': True,
    'dual_averaging': True
})

delta = np.array((0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95))
replications = 10
lag = 100

run_experiment(U, grad_U, start_position, iter, "mvn", delta, replications, dim, extra_pars, lag)

print("Doing logreg experiment")
log_reg = logReg()
U, grad_U = log_reg.params()

dim = 25
start_position = np.repeat(0, dim)
iter = 100
replications = 5
lag = 40
extra_pars["adapt_epsilon"] = 20
extra_pars["adapt_L"] = 20

run_experiment(U, grad_U, start_position, iter, "log_reg", delta, replications, dim, extra_pars, lag)

print("Doing stochastic volatility experiment")
stoch_vol = stochVolatility(obs=200)
U, grad_U = stoch_vol.params()

dim = 200 + 3
start_position = np.random.normal(size=dim)
iter = 500
replications = 5
lag = 40
run_experiment(U, grad_U, start_position, iter, "sv", delta, replications, dim, extra_pars, lag)

print("Doing banana experiment")
B = 0.1
dims = np.array([2, 10, 20])
iter = 1000
replications = 10
lag = 100
extra_pars["adapt_epsilon"] = 100
extra_pars["adapt_L"] = 100

for d in dims:
    banana = banana(B=B, dims=d)
    U, grad_U = banana.params()

    start_position = np.repeat(0, d)
    run_experiment(U, grad_U, start_position, iter, "banana", delta, replications, d, extra_pars, lag)
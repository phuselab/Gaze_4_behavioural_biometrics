import theano.tensor as tt
import pymc3 as pm
from pymc3.distributions import distribution, multivariate, continuous

class Mv_EulerMaruyama(distribution.Continuous):
    """
    Stochastic differential equation discretized with the Euler-Maruyama method.
    Parameters
    ----------
    dt : float
        time step of discretization
    sde_fn : callable
        function returning the drift and diffusion coefficients of SDE
    sde_pars : tuple
        parameters of the SDE, passed as *args to sde_fn
    """
    def __init__(self, dt, sde_fn, sde_pars, *args, **kwds):
        super(Mv_EulerMaruyama, self).__init__(*args, **kwds)
        self.dt = dt = tt.as_tensor_variable(dt)
        self.sde_fn = sde_fn
        self.sde_pars = sde_pars

    def logp(self, x):
        xt = x[:-1,:]
        f, g = self.sde_fn(x[:-1,:], *self.sde_pars)
        mu = xt + self.dt * f
        cov = self.dt * g
        #cov = tt.sqrt(self.dt) * g
        #sd = extract_diag(cov)
        #print(sd.tag.test_value.shape)
        #print(mu.tag.test_value.shape)
        #print(cov.tag.test_value.shape)
        #print(x[1:,:].tag.test_value.shape)
        #input('pause')
        #res = pm.MvNormal.dist(mu=mu, cov=cov).logp(x[:,1:])
        res = pm.MvNormal.dist(mu=mu, cov=cov).logp(x[1:,:])
        #res = pm.Normal.dist(mu=mu, sd=sd).logp(x[1:,:])
        return tt.sum(res)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        dt = dist.dt
        name = r'\text{%s}' % name
        return r'${} \sim \text{EulerMaruyama}(\mathit{{dt}}={})$'.format(name,get_variable_name(dt))


'''class vanilla_EulerMaruyama(distribution.Continuous):
    """
    Stochastic differential equation discretized with the Euler-Maruyama method.
    Parameters
    ----------
    dt : float
        time step of discretization
    sde_fn : callable
        function returning the drift and diffusion coefficients of SDE
    sde_pars : tuple
        parameters of the SDE, passed as *args to sde_fn
    """
    def __init__(self, dt, sde_fn, sde_pars, *args, **kwds):
        super(vanilla_EulerMaruyama, self).__init__(*args, **kwds)
        self.dt = dt = tt.as_tensor_variable(dt)
        self.sde_fn = sde_fn
        self.sde_pars = sde_pars

    def logp(self, x):
        xt = x[:-1]
        f, g = self.sde_fn(x[:-1], *self.sde_pars)
        mu = xt + self.dt * f
        sd = tt.sqrt(self.dt) * g
        #print(type(mu))
        #print(mu.tag.test_value.shape)
        #print(sd.tag.test_value.shape)
        #print(x[1:].tag.test_value.shape)
        return tt.sum(pm.Normal.dist(mu=mu, sd=sd).logp(x[1:]))

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        dt = dist.dt
        name = r'\text{%s}' % name
        return r'${} \sim \text{EulerMaruyama}(\mathit{{dt}}={})$'.format(name, get_variable_name(dt))


class my_MvStudentTRandomWalk(distribution.Continuous):
    """
    Multivariate Random Walk with StudentT innovations

    Parameters
    ----------
    nu : degrees of freedom
    mu : tensor
        innovation drift, defaults to 0.0
    cov : tensor
        pos def matrix, innovation covariance matrix
    tau : tensor
        pos def matrix, inverse covariance matrix
    chol : tensor
        Cholesky decomposition of covariance matrix
    init : distribution
        distribution for initial value (Defaults to Flat())
    """
    def __init__(self, nu=5., mu=0., Sigma=None, cov=None, tau=None, chol=None, lower=True, init=continuous.Flat.dist(), *args, **kwargs):
        super(my_MvStudentTRandomWalk, self).__init__(*args, **kwargs)

        self.init = init
        self.innovArgs = (Sigma, mu, cov, tau, chol, lower)
        self.nu = tt.as_tensor_variable(nu)
        self.innov = multivariate.MvStudentT.dist(self.nu, *self.innovArgs)
        self.mean = tt.as_tensor_variable(0.)

    def logp(self, x):
        x_im1 = x[:-1]
        x_i = x[1:]

        return self.init.logp_sum(x[0]) + self.innov.logp_sum(x_i - x_im1)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        nu = dist.innov.nu
        mu = dist.innov.mu
        cov = dist.innov.cov
        name = r'\text{%s}' % name
        return r'${} \sim \text{MvStudentTRandomWalk}(\mathit{{nu}}={},~\mathit{{mu}}={},~\mathit{{cov}}={})$'.format(name,
                                                get_variable_name(nu),
                                                get_variable_name(mu),
                                                get_variable_name(cov))


class my_StudentTRandomWalk(distribution.Continuous):
    """
    Random Walk with Normal innovations
    Parameters
    ----------
    mu: tensor
        innovation drift, defaults to 0.0
    sd : tensor
        sd > 0, innovation standard deviation (only required if tau is not specified)
    tau : tensor
        tau > 0, innovation precision (only required if sd is not specified)
    init : distribution
        distribution for initial value (Defaults to Flat())
    """

    def __init__(self, nu, tau=None, init=continuous.Flat.dist(), sd=None, mu=0., *args, **kwargs):
        super(my_StudentTRandomWalk, self).__init__(*args, **kwargs)
        #tau, sd = get_tau_sd(tau=tau, sd=sd)
        tau, sd = get_tau_sd(tau=tau, sigma=sd)
        self.tau = tau = tt.as_tensor_variable(tau)
        self.sd = sd = tt.as_tensor_variable(sd)
        self.mu = mu = tt.as_tensor_variable(mu)
        self.nu = tt.as_tensor_variable(nu)
        self.init = init
        self.mean = tt.as_tensor_variable(0.)

    def logp(self, x):
        tau = self.tau
        sd = self.sd
        mu = self.mu
        init = self.init
        nu = self.nu

        x_im1 = x[:-1]
        x_i = x[1:]

        innov_like = pm.StudentT.dist(nu=nu, mu=x_im1 + mu, sd=sd).logp(x_i)
        return init.logp(x[0]) + tt.sum(innov_like)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        mu = dist.mu
        sd = dist.sd
        name = r'\text{%s}' % name
        return r'${} \sim \text{{GaussianRandomWalk}}(\mathit{{mu}}={},~\mathit{{sd}}={})$'.format(name,
                                                get_variable_name(mu),
get_variable_name(sd))
'''
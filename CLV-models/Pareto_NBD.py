__author__ = 'Alex'

from numpy import array, log, logaddexp, ones
from scipy.special import hyp2f1, gammaln
from scipy.optimize import differential_evolution
from scipy.misc import logsumexp
from utils import *
import sympy.mpmath as mp


class ParetoNBD:
    def __init__(self, pars=None, penalty=0.):
        self.penalty = penalty
        self.pars = pars

    @staticmethod
    def _log_a_0(r, alpha, s, beta, freq, rec, age):
        min_ab, max_ab, t = (alpha, beta, r + freq) if alpha < beta else (beta, alpha, s + 1)
        abs_ab = max_ab - min_ab
        rsx = r + s + freq
        p_1, q_1 = hyp2f1(rsx, t, rsx + 1., abs_ab / (max_ab + rec)), (max_ab + rec)
        p_2, q_2 = hyp2f1(rsx, t, rsx + 1., abs_ab / (max_ab + age)), (max_ab + age)
        sign = ones(len(freq))
        return logsumexp([log(p_1) + rsx * log(q_2), log(p_2) + rsx * log(q_1)],
                         axis=0, b=[sign, -sign]) - rsx * log(q_1 * q_2)

    def pareto_ndb_ll(self, pars, freq, rec, age):
        r, alpha, s, beta = pars
        rsx = r + s + freq
        log_a_0 = self._log_a_0(r, alpha, s, beta, freq, rec, age)
        a_1 = gammaln(r + freq) - gammaln(r) + r * log(alpha) + s * log(beta)
        a_2 = logaddexp(-(r + freq) * log(alpha + age) - s * log(beta + age), log(s) + log_a_0 - log(rsx))
        return -(a_1 + a_2).sum() + self.penalty * log(pars).sum()

    def fit(self, freq, rec, age):
        freq, rec, age = check_inputs(freq, rec, age)
        result = differential_evolution(self.pareto_ndb_ll, [[1e-3, 20]] * 4, (freq, rec, age), popsize=50, tol=1e-3)
        print(result)
        self.pars = result.x
        return self

    def p_alive(self, freq, rec, age):
        freq, rec, age = check_inputs(freq, rec, age)
        r, alpha, s, beta = self.pars
        log_a_0 = self._log_a_0(r, alpha, s, beta, freq, rec, age)

        def precise(freq_, age_, log_a_0_):
            return float(mp.power(alpha + age_, r + freq_) * mp.exp(log_a_0_))

        p = array(list(map(precise, freq, age, log_a_0)))
        return 1. / (1. + (s / (r + s + freq)) * (beta + age) ** s * p)
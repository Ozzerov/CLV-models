__author__ = 'Alex'

from numpy import log, logaddexp
from scipy.special import hyp2f1, gammaln


def pareto_ndb_ll(pars, freq, rec, age, penalty):
    r, alpha, s, beta = pars
    x = freq
    i = alpha > beta
    max_ab = alpha if i else beta
    min_ab = beta if i else alpha
    abs_ab = max_ab - min_ab
    t = s + 1 if i else r + x
    rsx = r + s + x
    a0 = (hyp2f1(rsx, t, rsx + 1., abs_ab / (max_ab + rec)) / (max_ab + rec) ** rsx -
          hyp2f1(rsx, t, rsx + 1., abs_ab / (max_ab + age)) / (max_ab + age) ** rsx)
    a1 = gammaln(r + x) - gammaln(r) + r * log(alpha) + s * log(beta)
    a2 = logaddexp(-(r + x) * log(alpha + age) - s * log(beta + age), log(s) + log(a0) - log(rsx))
    return -(a1 + a2).sum() + penalty * log(pars).sum()

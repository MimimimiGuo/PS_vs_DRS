import numpy as np


class Parameters(object):

    def __init__(self,
                 n=None,
                 a0_treat=None,
                 b0_outcome=None,
                 b_treat=None,
                 n_confound_norm=None,
                 n_confound_binm=None,
                 n_confound_gamm=None,
                 n_instruml_binm=None,
                 n_risk_binm=None,
                 n_trials=None,
                 interaction_percent=None,
                 **kwargs):

        self.n = 500  # number of observations per variable/ number of patients
        self.a0_treat = -1.3
        self.b0_outcome = -0.8
        self.b_treat = np.log(2)
        self.n_confound_norm = 25 * 1  # number of normal distributed confounders
        self.n_confound_binm = 25 * 1  # number of binomial distributed confounders
        self.n_confound_gamm = 30 * 1  # number of gamma distributed confounders
        self.n_instruml_binm = 5 * 1  # number of binomial distributed instrumental variables
        self.n_risk_binm = 1 * 1  # number of binomial distributed risk factors
        self.n_trials = 1
        self.interaction_percent = 0.3
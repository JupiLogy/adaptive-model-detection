import numpy as np
from scipy.special import chdtrc
from sklearn.cluster import dbscan

class amd():
    def __init__(self, default_state, a, o, h_len=40, dead_val=10):
        """
            default_state - TODO
            a - number of actions available in environment
            o - number of observations possible in environment
        """
        self.h = []  # History of [a, o] pairs
        # Ignore the very first state as it has no corresponding a, o pair
        self.s = np.empty([0, len(default_state), len(default_state[0])])  # History of PSR states
        self.observed = []  # Observed state cluster means
        self.s_labels = np.array([0])  # Stores cluster of each state (-1 means unassigned)
        self.a = a
        self.o = o
        self.dead = dead_val
        self.h_len = h_len  # Max length of stored history
        self.act = np.zeros((1, self.a, self.o))  # Tallies actual observed frequencies for each clust
        self.exp = np.zeros((1, self.a, self.o))  # Stores expected observed frequencies for each clust
        self.p_val = 1 # p-value of observed data matching agent's predictions

    def step(self, state, a, o):
        """
            state should be new agent internal state after taking
            action a and observing observation o.
            state should contain probability distribution of next observation.
        """
        try:
            if state.type() == 'torch.FloatTensor':
                state = state.detach().numpy()
        except AttributeError:
            pass
        if self.h and len(self.s) > self.h_len:
            self.h = self.h[1:]
            self.s = self.s[1:]
            self.s_labels = self.s_labels[1:]
        self.s = np.append(self.s, [state], axis=0)
        self.h.append([a, o])
        self.s_labels = dbscan([state.flatten() for state in self.s])[1]
        self.observed = np.zeros((max(self.s_labels) + 1, self.a, self.o))
        for i in range(max(self.s_labels) + 1):
            state_indices = [j for j in range(len(self.s)) if self.s_labels[j] == i]
            clust_mean = np.mean([self.s[j] for j in state_indices], axis=0)
            self.observed[i] = clust_mean
            self.update_act()
            self.update_exp()

        self.calc_chisquare()

    def update_act(self):
        self.act = np.zeros((len(self.observed), self.a, self.o))
        for i in range(len(self.observed)):
            sinds = [j for j in range(len(self.s) - 1) if self.s_labels[j] == i]
            for ind in sinds:
                self.act[i, self.h[ind + 1][0], self.h[ind + 1][1]] += 1

    def update_exp(self):
        self.exp = np.zeros((len(self.observed), self.a, self.o))
        for i in range(len(self.observed)):
            sinds = [j for j in range(len(self.s) - 1) if self.s_labels[j] == i]
            for a in range(self.a):
                actia = 0
                for ind in sinds:
                    if self.h[ind + 1][0] == a:
                        actia += 1
                for o in range(self.o):
                    self.exp[i, a, o] = self.observed[i][a][o] * actia

        self.exp[self.exp < 1e-8] = 0

    def remove_test(self, s_ind):
        del self.observed[s_ind]
        self.exp = np.delete(self.exp, s_ind, axis=0)
        self.act = np.delete(self.act, s_ind, axis=0)
        self.s_labels[self.s_labels > s_ind] -= 1

    def calc_chisquare(self):
        # chisquare.
        obs = np.reshape(self.act, -1)
        exp = np.reshape(self.exp, -1)
        oesd = np.zeros(len(obs))
        for i in range(len(obs)):
            if obs[i] <= 1e-8 >= exp[i]:
                oesd[i] = 0
            elif exp[i] <= 1e-8:
                oesd[i] = self.dead * obs[i] ** 2
            else:
                oesd[i] = (obs[i] - exp[i]) ** 2 / exp[i]
        stat = np.sum(oesd)
        dof = self.get_num_params()
        self.p_val = chdtrc(dof, stat)

    def get_num_params(self):
        tmp = 0
        for s in range(len(self.observed)):
            for a in range(self.a):
                if sum(self.exp[s, a]) > 0:
                    tmp += 1
        k = (tmp) * (self.o - 1)
        k = (tmp) * (self.o)
        return k

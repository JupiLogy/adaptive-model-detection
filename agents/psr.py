import numpy as np
import math

"""
    model = constraned_grad(a, o, n) sets up the model with:
        - a choices of action
        - o possible observations
        - n stored history length
        - (see __init__ function for more parameter options)

    model.step(a, o) indicates that:
        - a is the next action taken
        - o is the next observation observed.

    You must step the model at every timestep in order to get an accurate model!
"""


class const_grad():
    """
        Constrained Gradient Algorithm
        ------------------------------
        a : number of actions
        o : number of observations
        n : how frequently the algorithm searches for a better set of tests.
        cond : condition threshold. Higher means more tests will be gathered,
               and the algorithm will be stricter. Default value is 10.
        cumulative_discovery : if this is set to False, existing found tests
                               will be discarded and new tests will be found.
                               Otherwise, the old tests will be used as a
                               starting point for finding new tests. Default
                               is True.
        learning_rate : set to -1 to allow decaying learning rate. Otherwise,
                        learning rate can be in the range 0 < alph <= 1
        halving_interval : if learning_rate is set to -1, then halving interval
                           is the half-life of the learning rate, in timesteps.
    """

    def __init__(
        self,
        a,
        o,
        n=500,
        h_len=1000,
        core_t=[np.empty((0, 2), dtype=np.int8)],
        cond=10,
        cumulative_discovery=True,
        learning_rate=0.99999,
        decay=0.9,
        halving_interval=100000,
    ):
        np.set_printoptions(edgeitems=30, linewidth=100000,
            formatter=dict(float=lambda x: "%.3g" % x))
        self.a = a
        self.o = o
        self.n = n
        self.h_len = h_len
        self.h = []
        self.core_t = core_t
        self.get_norm_tests()
        self.p = np.full((len(self.norm_t), 1), 1e-1)
        # Previous TimeStep and TimeStep, in the range of n. Cyclical.
        self.pts = -1
        self.ts = 0
        self.normalise()
        # Stores updated tests from each timestep, used in update_from_future
        self.last_updated = []
        self.consec = True  # If timesteps were skipped, this becomes False

        self.c = cond
        self.cumulative_discovery = cumulative_discovery

        self.decay = decay
        self.alph = learning_rate
        assert learning_rate < 1
        if halving_interval == -1:
            self.h_interval = None
        else:
            self.h_interval = halving_interval

    def step(self, a, o):
        """
            Step:
            Adds action a and observation o to history, and rotates history.
            Updates probability matrix.
            If it's time to discover, runs discovery step.
        """

        if not self.consec and len(self.h) > 0:
            for l in range(min(math.floor(len(self.p[0]) / 2), math.floor(len(self.p[0]) - self.h_len / 2))):  # Don't cut off more than half of history!
                try:
                    if np.allclose(self.p2, self.p[: len(self.core_t), -l], atol=5e-2):
                        len_del_hist = l
                        break
                except:
                    import pdb; pdb.set_trace()
            else:
                self.no_learn_step(a, o)
                return
            if len_del_hist > 0:
                self.h = self.h[: -len_del_hist]
                self.p = self.p[:, : -len_del_hist]
        self.consec = True

        # incrementing timestep count
        self.pts = self.ts
        # if self.ts < self.n - 1:
        self.ts += 1
        # else:
        #     self.ts = 0

        # We will not update if length of longest norm test has increased.
        update = True

        if self.h_interval:
            if self.ts % self.h_interval == 0:
                self.alph = max(self.decay*self.alph, 0.01)

        if (
            # self.p.shape[1] < max([len(test) for test in self.core_t])
            # and
            len(self.h) < max([len(test) for test in self.core_t]) + 1
        ):
            # future buffer on early timesteps, allowing future updates
            update = False

        # discovering tests every self.n timesteps (plus len longest norm test)
        if (
            self.ts % self.n == max([len(test) for test in self.norm_t]) + 1
            and self.p.shape[1] >= self.h_len
        ):
            # keeping track of length of longest norm test
            # to allow for future buffer
            old_len_norm_t = max([len(test) for test in self.norm_t])
            self.discover()
            new_len_norm_t = max([len(test) for test in self.norm_t])
            if old_len_norm_t != new_len_norm_t or self.p.shape[1] < 2:
                update = False

        # updating model parameters
        if len(self.h) < self.h_len:
            # Can't update model parameters if we have no set of tests yet.
            self.h.append([a, o])
        else:
            self.h = self.h[1:]
            self.h.append([a, o])

        if update:
            self.update_p_matrix()
        else:
            self.last_updated = [test[1:] for test in self.last_updated]

    def no_learn_step(self, a, o):
        # Only updates current state. History and learning are frozen.
        if self.consec:
            self.psr_params = self.get_psr_params()
            # only calculate this on first step to reduce computation
            self.p2 = self.get_latest_state()
            # corresponds to the state before last input a, o
        self.consec = False
        self.p2 = normalise_generic(self.psr_params[1][a][o] @ np.transpose(self.p2))

    def get_est(self, a, o):
        # Like no_learn_step, but doesn't make changes to agent's state at all -
        # - good for getting next prediction
        if self.consec:
            self.psr_params = self.get_psr_params()
            temp_p = self.get_latest_state()
        else:
            temp_p = self.p2
        return (self.psr_params[1][a][o] @ np.transpose(temp_p))[0]

    def get_latest_state(self):
        # As the algorithm uses future observations to determine current state,
        # the most recent state is actually for a past time.
        # This returns the predicted state for the most recent timestep.
        buffer_h = self.h[-self.norm_t[-1].shape[0]:]
        s = self.p[: len(self.core_t), -1]
        for step in buffer_h:
            try:
                s = self.psr_params[1][step[0]][step[1]] @ s
                s = normalise_generic(s)
            except:
                self.psr_params = self.get_psr_params()
                s = self.psr_params[1][step[0]][step[1]] @ s
                s = normalise_generic(s)
        return s

    def update_p_matrix(self):
        # For this part, since our self.p is the transpose of the one
        # described by McCallum in the literature, therefore
        # for these equations we have transposed self.p when McCallum
        # did not, and vice versa.
        core_inds = [np_in_list(t, self.norm_t)[1] for t in self.core_t]
        a, o = self.h[-self.norm_t[-1].shape[0] - 1]

        try:
            A = (
                np.linalg.inv(
                    (self.p[core_inds] @ np.transpose(self.p[core_inds]))
                    + np.eye(self.p[core_inds].shape[0]) * (10 ** (-4))
                )
                @ self.p[core_inds]
            )
        except:
            # Just in the tiny chance that the matrix in the try block is singular
            A = (
                np.linalg.inv(
                    (self.p[core_inds] @ np.transpose(self.p[core_inds]))
                )
                @ self.p[core_inds]
            )

        # t_1 : Set of tests which begin with a, o, where the remainder of the test is also a test
        t_1 = [
            t
            for t in self.norm_t
            if np_in_list(np.append([[a, o]], t, axis=0), self.norm_t)[0]
        ]
        t_2 = [t for t in self.norm_t if not np_in_list(t, t_1)[0]]

        ao_ind = np_in_list(np.array([[a, o]]), self.norm_t)[1]
        new_row = np.full(self.p.shape[0], 10 ** (-3))

        for t in t_1:
            t_ind = np_in_list(t, self.norm_t)[1]
            aot_ind = np_in_list(np.append([[a, o]], t, axis=0), self.norm_t)[1]
            new_row[t_ind] = self.p[aot_ind, -1] / self.p[ao_ind, -1]

        for t in t_2:
            t_ind = np_in_list(t, self.norm_t)[1]
            m_t = A @ np.transpose(self.p[t_ind])
            new_row[t_ind] = new_row[core_inds] @ np.transpose(m_t)

        if self.p.shape[1] == self.h_len:
            self.p[:, :-1] = self.p[:, 1:]
            self.p[:, -1] = new_row
        else:
            self.p = np.append(self.p, np.transpose([new_row]), axis=1)

        self.normalise()

        self.update_from_future()

    def normalise(self, i=-1):
        new_row = np.zeros(len(self.norm_t))
        np.nan_to_num(self.p)
        clip(self.p, min=1e-10)
        for j in range(max([len(test) for test in self.norm_t]) + 1):
            if j == 0:
                new_row[0] = 1
            else:
                for t_ind in range(len(self.norm_t)):
                    if len(self.norm_t[t_ind]) == j:
                        # get sibling indices
                        sibs = self.get_sibs(t_ind)
                        denom = np.sum(self.p[sibs, i])
                        par_ind = self.get_par(t_ind)
                        if denom > 0:
                            new_row[t_ind] = self.p[t_ind, i] * new_row[par_ind] / denom
                        else:
                            # Denominator will be 0 if siblings have chance 0.
                            new_row[sibs] = new_row[par_ind] / self.o
                        try:
                            assert new_row[t_ind] <= 1
                        except:
                            print(new_row)
                            print(self.p)
                            import pdb; pdb.set_trace()
        self.p[:, i] = new_row

    def update_from_future(self):
        new_updated = []
        max_len = max([len(test) for test in self.norm_t])
        for i in range(1, max_len + 1):
            # i is length of test
            if max_len - i == 0:
                len_i_hist = self.h[-max_len:]
            else:
                len_i_hist = self.h[-max_len : i - max_len]

            len_i_tests = [test for test in self.norm_t if test.shape[0] == i]
            for t in len_i_tests:
                if (
                    np.array_equal(t, len_i_hist)
                    and not np_in_list(t, [test[1:] for test in self.last_updated])[0]
                ):
                    new_updated.append(t)
                    t_ind = np_in_list(t, self.norm_t)[1]
                    par_ind = np_in_list(t[:-1], self.norm_t)[1]
                    self.p[t_ind, -1] +=\
                        (
                            self.p[par_ind, -1] * (
                                (1 - self.alph) * self.p[t_ind, -1] / self.p[par_ind, -1]
                                + self.alph
                            ) - self.p[t_ind, -1]
                        ) / (
                            1
                            + (self.alph - 1) * self.p[t_ind, -1] / self.p[par_ind, -1]
                            - self.alph
                        )
                    self.normalise()
                    break
        self.last_updated = new_updated.copy()

    def discover(self):
        if not self.cumulative_discovery:
            self.core_t = [np.empty((0, 2))]

        while True:
            S = []
            for test in self.core_t:
                for a in range(self.a):
                    for o in range(self.o):
                        if np.array_equal(test, np.empty((0, 2))):
                            S.append(np.array([[a, o]]))
                        else:
                            if not np_in_list(np.append([[a, o]], test, axis=0), S)[0]:
                                S.append(np.append([[a, o]], test, axis=0))
            t = -1
            cond = float("inf")
            for i in range(len(S)):
                if not np_in_list(S[i], self.core_t)[0]:
                    x = np.linalg.cond(self.get_p_matrix(self.core_t + [S[i]]))
                    if x < cond:
                        cond = x
                        t = i
            if cond < self.c:
                self.core_t.append(S[t])
            else:
                break
        old_norm_tests = np.copy(self.norm_t)
        self.get_norm_tests()
        self.change_p_tests(old_norm_tests)

    def change_p_tests(self, old_norm_tests):
        """
            Generates a probability matrix given existing data in self.p
        """
        new_p = np.full((len(self.norm_t), self.p.shape[1]), 10 ** (-3))
        for old_ind in range(len(old_norm_tests)):
            new_ind = np_in_list(old_norm_tests[old_ind], self.norm_t)[1]
            new_p[new_ind] = np.copy(self.p[old_ind])
        self.p = np.copy(new_p)
        self.normalise()

    def get_par(self, t_ind):
        par_ind = np_in_list(self.norm_t[t_ind][:-1], self.norm_t)[1]
        return par_ind

    def get_sibs(self, t_ind):
        a = self.norm_t[t_ind][-1, 0]
        par_ind = self.get_par(t_ind)
        sibs = np.zeros(self.o, dtype=np.int16)
        for o in range(self.o):
            sib = np.append(self.norm_t[par_ind], [[a, o]], axis=0)
            sib_ind = np_in_list(sib, self.norm_t)[1]
            sibs[o] = sib_ind
        return sibs

    def get_p_matrix(self, tests):
        inds = [np_in_list(x, self.norm_t)[1] for x in tests]
        return np.array([self.p[ind] for ind in inds])

    def get_norm_tests(self):
        self.norm_t = self.core_t.copy()

        # Get one step extensions
        for test in self.core_t:
            for a in range(self.a):
                for o in range(self.o):
                    aot = np.append([[a, o]], test, axis=0)
                    if not np_in_list(aot, self.norm_t)[0]:
                        self.norm_t.append(aot)
        old_norm_t = self.core_t.copy()

        while old_norm_t != self.norm_t:
            old_norm_t = self.norm_t.copy()
            for t in old_norm_t:
                # Add parent
                parent = t[:-1]
                if not np_in_list(parent, self.norm_t)[0]:
                    self.norm_t.append(parent)

                # Add siblings
                if t != np.empty((0, 2)):
                    a = t[-1][0]
                    for o in range(self.o):
                        sibling = np.append(parent, [[a, o]], axis=0)
                        if not np_in_list(sibling, self.norm_t)[0]:
                            self.norm_t.append(sibling)

    def get_psr_params(self):
        core_inds = [np_in_list(t, self.norm_t)[1] for t in self.core_t]

        state = np.sum(self.p[core_inds], axis=1) / len(self.p[0])

        A = (
            np.linalg.inv(
                (self.p[core_inds] @ np.transpose(self.p[core_inds]))
                + np.eye(self.p[core_inds].shape[0]) * (10 ** (-4))
            )
            @ self.p[core_inds]
        )

        M = []
        os_ex = []
        for a in range(self.a):
            part1 = []
            M1 = []
            for o in range(self.o):
                part2 = []
                M2 = []
                for test_ind in core_inds:
                    if test_ind == 0:
                        part2.append(np.array([[a, o]]))
                    else:
                        part2.append(np.append([[a, o]], self.norm_t[test_ind], axis=0))
                    M2.append(A @ np.transpose(self.p[np_in_list(part2[-1], self.norm_t)[1]]))
                part1.append(part2)
                M1.append(M2)
            os_ex.append(part1)
            M.append(M1)
        M = np.array(M)
        for test_ind in core_inds:
            if test_ind != 0 and not np_in_list(self.norm_t[test_ind], flatten(os_ex))[0]:
                os_ex.append(self.norm_t[test_ind])
        return state, M, A, os_ex

    def get_aoinds(self):
        aoinds = np.zeros((self.a, self.o), dtype=int)
        for a in range(self.a):
            for o in range(self.o):
                aoinds[a, o] = np_in_list([[a, o]], self.norm_t)[1]
        return aoinds


def np_in_list(arr, lis):
    a = [np.array_equal(arr, x) for x in lis]
    if True in a and len(lis) > 0:
        return True, a.index(True)
    else:
        return False, 0


def flatten(lis):
    out = []
    for item in lis:
        if isinstance(item, list):
            for x in flatten(item):
                out.append(x)
        else:
            out.append(item)
    return out


def clip(arr, min=-float("Inf"), max=float("Inf")):
    arr[arr < min] = min
    arr[arr > max] = max


def normalise_generic(arr):
    if arr.min() < 0:
        arr -= arr.min()
    bigg = max(arr)
    arr = arr/bigg
    return arr

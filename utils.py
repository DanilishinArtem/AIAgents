import numpy as np
import math
from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
from torch import nn

# ------------------ New realization according to the new theory and full likelihood ------------------



class HMMTwoAgentsFeatures:
    def __init__(self, n_states: int, n_actions: int, n_features: int,
                 rand_state: int | None = None, psi: float = 1e-6,
                 l2_eta: float = 1e-4):
        self.EPS = 1e-12
        self.K = n_states
        self.A = n_actions
        self.F = n_features
        self.rng = np.random.default_rng(rand_state)

        # initial distribution of the states: pi(s)
        self.pi = self.rng.random(self.K); self.pi /= self.pi.sum()
        # initial probabilities of the transition matrix: П(s,s')
        self.Pi = self.rng.random((self.K, self.K)); self.Pi /= self.Pi.sum(axis=1, keepdims=True)

        # parameters for 𝑝(𝑧_(1,𝑡),𝑧_(2,𝑡) | 𝑠_𝑡=𝑠,Θ)∝exp⁡(𝑒(𝑧_(1,𝑡) )^𝑇 𝜃_𝑠^((1) )+𝑒(𝑧_(2,𝑡) )^𝑇 𝜃_𝑠^((2) )+𝑒(𝑧_(1,𝑡) )^𝑇 𝑊_𝑠 𝑒(𝑧_(2,𝑡) ))
        self.theta1 = self.rng.normal(scale=0.1, size=(self.K, self.A))
        self.theta2 = self.rng.normal(scale=0.1, size=(self.K, self.A))
        self.W = self.rng.normal(scale=0.1, size=(self.K, self.A, self.A))

        # parameters for 𝑝(𝑥_(𝑖,𝑡)│𝑧_(𝑖,𝑡)=𝑎,𝑠_𝑡=𝑠;Φ)=𝑁(𝑥_(𝑖,𝑡);𝜇_(𝑠,𝑖,𝑎),𝑑𝑖𝑎𝑔(𝜎_(𝑠,𝑖,𝑎)^2 ))
        # K - states, 2 - agents, A - actions, F - features
        self.phi_mean = self.rng.normal(scale=0.5, size=(self.K, 2, self.A, self.F))
        self.phi_var = np.ones((self.K, 2, self.A, self.F))

        # parameters of regularization
        self.psi = float(psi)      # For π, Π
        self.l2_eta = float(l2_eta)  # L2 for η

    # Safe version of the logsum exponent function 
    # log⁡〖(∑8_𝑖 𝑒^(𝑥_𝑖 ) )=𝑚+log⁡(∑_𝑖 〖𝑒^(𝑥_𝑖 )−𝑚〗),𝑚=max_𝑖⁡〖𝑥_𝑖 〗 〗
    def logsumexp(self, x, axis=None):
        m = np.max(x, axis=axis, keepdims=True)
        return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)

    # ---------- Additional probabilities ----------
    def _logp_z_given_s(self, s: int) -> np.ndarray:
        # log⁡〖(𝑝(𝑧_1=𝑎,𝑧_2=𝑏|𝑠,𝜂))=𝜃_(𝑠,𝑎)^((1))+𝜃_(𝑠,𝑏)^((2))+𝑊_(𝑠,𝑎𝑏)−log⁡(∑𝜃_(𝑠,𝑎′)^((1) )+𝜃_(𝑠,𝑏′)^((2) )+𝑊_(𝑠,𝑎′𝑏′) ) 〗
        scores = self.theta1[s][:, None] + self.theta2[s][None, :] + self.W[s]  # (A,A)
        logZ = self.logsumexp(scores.ravel())
        return scores - logZ

    @staticmethod
    def _gaussian_logpdf_diag(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
        # x, mean, var: (F,)
        var_safe = np.maximum(var, 1e-8)
        return -0.5 * (np.sum(np.log(2*np.pi*var_safe) + (x-mean)**2 / var_safe))

    def _log_emission(self, s: int, z1: int, z2: int, x1: np.ndarray, x2: np.ndarray) -> float:
        # p(observations|hidden_states), where
        # observations: z1, z2, x1, x2
        logPz = self._logp_z_given_s(s)[z1, z2]
        logPx1 = self._gaussian_logpdf_diag(x1, self.phi_mean[s,0,z1], self.phi_var[s,0,z1])
        logPx2 = self._gaussian_logpdf_diag(x2, self.phi_mean[s,1,z2], self.phi_var[s,1,z2])
        return logPz + logPx1 + logPx2

    # ---------- forward-backward ----------
    def _forward_backward(self, z_seq: np.ndarray, x_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        # part of calculation b_t(s) = p(z1, z2 | s) * p(x_1|z_1,s) * p(x_2|z_2,s)
        T = z_seq.shape[0]
        logB = np.empty((T, self.K))
        for t in range(T):
            z1, z2 = z_seq[t]
            x1, x2 = x_seq[t,0], x_seq[t,1]
            for s in range(self.K):
                logB[t, s] = self._log_emission(s, z1, z2, x1, x2)


        # forward in log-domain with scaling
        # 𝛼_(𝑡+1) (𝑠^′ )=(∑_𝑠▒〖𝛼_𝑡 (𝑠)Π(𝑠^′ |𝑠) 〗) 𝑏_(𝑡+1) (𝑠^′ )
        log_alpha = np.empty((T, self.K))
        log_alpha[0] = np.log(np.maximum(self.pi, self.EPS)) + logB[0]
        c0 = self.logsumexp(log_alpha[0])
        log_alpha[0] -= c0
        log_scales = np.empty(T); log_scales[0] = c0
        logPi = np.log(np.maximum(self.Pi, self.EPS))
        for t in range(1, T):
            # log_alpha[t,j] = logB[t,j] + logsumexp_i (log_alpha[t-1,i] + logPi[i,j])
            log_alpha[t] = logB[t] + self.logsumexp(log_alpha[t-1][:, None] + logPi, axis=0)
            ct = self.logsumexp(log_alpha[t])
            log_alpha[t] -= ct
            log_scales[t] = ct
        loglik = np.sum(log_scales)

        # backward in log-domain with scaling
        # 𝛽_𝑡 (𝑠)=∑_𝑠′▒〖Π(𝑠^′ |𝑠) 𝑏_(𝑡+1) (𝑠^′ ) 𝛽_(𝑡+1) (𝑠^′ ) 〗
        log_beta = np.empty((T, self.K))
        log_beta[-1] = 0.0
        for t in range(T-2, -1, -1):
            # log_beta[t,i] = logsumexp_j ( logPi[i,j] + logB[t+1,j] + log_beta[t+1,j] )
            log_beta[t] = self.logsumexp(logPi + (logB[t+1] + log_beta[t+1])[None, :], axis=1)
            # scale to keep consistency (subtract self.logsumexp for numerical stability)
            c = self.logsumexp(log_beta[t])
            log_beta[t] -= c

        # gamma
        # 𝛾_𝑡 (𝑠)=(𝛼_𝑡 (𝑠) 𝛽_𝑡 (𝑠))/(∑2_𝑢▒〖𝛼_𝑡 (𝑢) 〗 𝛽_𝑡 (𝑢) )
        log_gamma = log_alpha + log_beta
        # normalize per t
        log_gamma = (log_gamma.T - self.logsumexp(log_gamma, axis=1)).T
        gamma = np.exp(log_gamma)

        # xi
        # 𝜉_𝑡 (𝑠,𝑠^′ )=(𝛼_𝑡 (𝑠)Π(𝑠^′│𝑠) 𝑏_(𝑡+1) (𝑠^′ ) 𝛽_(𝑡+1) (𝑠^′ ))/(∑2_(𝑢,𝑣)▒〖𝛼_𝑡 (𝑢)Π(𝑣│𝑢) 𝑏_(𝑡+1) (𝑣) 𝛽_(𝑡+1) (𝑣) 〗)
        xi = np.zeros((T-1, self.K, self.K))
        for t in range(T-1):
            # unnormalized in log
            M = (log_alpha[t][:, None] + logPi + logB[t+1][None, :] + log_beta[t+1][None, :])
            M -= self.logsumexp(M)
            xi[t] = np.exp(M)

        return gamma, xi, loglik

    # ---------- M-step: η (log-linear optimization) ----------
    def _optimize_eta_for_state(self, s: int, C_pair: np.ndarray, n_s: float, max_iter: int = 200, lr: float = 0.5) -> None:
        # C_pair: a matrix of counters (empirical statistics), for example, how many times 
        # the pair (z₁,z₂) appeared in the data when the hidden state was s
        if n_s <= 0:
            return
        th1 = self.theta1[s].copy()
        th2 = self.theta2[s].copy()
        W = self.W[s].copy()
        def pack(th1, th2, W):
            return th1, th2, W

        def objective_and_grad(th1, th2, W):
            scores = th1[:, None] + th2[None, :] + W           # (A,A)
            logZ = self.logsumexp(scores.ravel())
            P = np.exp(scores - logZ)                          # softmax (A,A)

            # obj: an empirical coincidence (how well does the model explain the data)
            obj = np.sum(C_pair * scores) - n_s * logZ
            # L2 regularization
            obj -= 0.5 * self.l2_eta * (np.sum(th1**2) + np.sum(th2**2) + np.sum(W**2))

            # grads (empirical - n_s * model) - λ * θ
            # gradient = empirical statistics − model−expected statistics - regularization
            G_pair = C_pair - n_s * P                          # (A,A)
            g_th1 = np.sum(G_pair, axis=1) - self.l2_eta * th1 # (A,)
            g_th2 = np.sum(G_pair, axis=0) - self.l2_eta * th2 # (A,)
            g_W   = G_pair - self.l2_eta * W                   # (A,A)
            return obj, g_th1, g_th2, g_W

        prev_obj, _, _, _ = objective_and_grad(th1, th2, W)
        for it in range(max_iter):
            obj, g1, g2, gW = objective_and_grad(th1, th2, W)
            # backtracking line search
            step = lr
            improved = False
            for _ in range(20):
                th1_new = th1 + step * g1
                th2_new = th2 + step * g2
                W_new   = W   + step * gW
                obj_new, _, _, _ = objective_and_grad(th1_new, th2_new, W_new)
                if obj_new >= obj - 1e-9:  # not became worse
                    th1, th2, W = th1_new, th2_new, W_new
                    prev_obj = obj_new
                    improved = True
                    break
                step *= 0.5
            if not improved:
                # less then epsilong, we can break it
                break
        self.theta1[s], self.theta2[s], self.W[s] = pack(th1, th2, W)

    # ---------- M-step: Φ (weighted gaussian MLE) ----------
    def _update_phi(self, stats: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]):
        K, A, F = self.K, self.A, self.F
        # sum accumulation
        sum_w = np.zeros((K, 2, A)) # accumulated weight (gamma)
        sum_x = np.zeros((K, 2, A, F)) # accumulated sum of observations x
        sum_x2 = np.zeros((K, 2, A, F)) # accumulated sum of observations x^2

        for gamma, z_seq, x_seq in stats:
            T = z_seq.shape[0]
            for t in range(T):
                z1, z2 = z_seq[t] # actions of agents
                x1, x2 = x_seq[t,0], x_seq[t,1] # observations of agents
                for s in range(K):
                    w = gamma[t, s] # probability p(s_t = s | datas)
                    if w <= 0: 
                        continue
                    # agent 1, action z1
                    sum_w[s,0,z1] += w # accumulated weight for agent 1
                    sum_x[s,0,z1] += w * x1 # accumulate statistic for mean for agent 1
                    sum_x2[s,0,z1] += w * (x1**2) # accumulate statistic for deviation for agent 1
                    # agent 2, action z2
                    sum_w[s,1,z2] += w # accumulated weight for agent 2
                    sum_x[s,1,z2] += w * x2 # accumulate satistic for mean for agent 2
                    sum_x2[s,1,z2] += w * (x2**2) # accumulate statistic for deviation for agent 2

        # update mean/var
        # 𝜇=(∑𝑤_𝑡 𝑥_𝑡)/(∑𝑤_𝑡 )
        mean = np.copy(self.phi_mean)
        # 𝜎^2=(∑𝑤_𝑡 𝑥_𝑡^2)/(∑𝑤_𝑡 )−𝜇^2
        var = np.copy(self.phi_var)
        for s in range(K):
            for i in range(2):
                for a in range(A):
                    w = sum_w[s,i,a]
                    if w > 0:
                        m = sum_x[s,i,a] / w
                        # deviation through coordinates
                        v = (sum_x2[s,i,a] / w) - (m**2)
                        v = np.maximum(v, 1e-6)
                        mean[s,i,a] = m
                        var[s,i,a] = v
        self.phi_mean = mean
        self.phi_var = var

    # ---------- total EM algorithm ----------
    def fit_em(self, sequences: List[Tuple[np.ndarray, np.ndarray]], n_iter: int = 50, tol: float = 1e-4, verbose: bool = True):
        prev_ll = -np.inf
        for it in range(1, n_iter+1):
            total_ll = 0.0
            # statistics
            N0 = np.zeros(self.K) # for initial probabilities π
            Nij = np.zeros((self.K, self.K)) # for transitions Π
            all_stats = []  # for Φ (mean/deviations of observations)
            # for η: accumulate for states C_pair[s, a, b]
            C_pair_all = np.zeros((self.K, self.A, self.A)) # for η (matrix of quantity of pairs actions of agens for given hidden state)
            n_s = np.zeros(self.K) # normalization (total mass in state s)

            # --- E-step ---
            for z_seq, x_seq in sequences:
                gamma, xi, ll = self._forward_backward(z_seq, x_seq)
                total_ll += ll
                N0 += gamma[0]
                Nij += xi.sum(axis=0)
                all_stats.append((gamma, z_seq, x_seq))
                # statistics for η
                T = z_seq.shape[0]
                for t in range(T):
                    a, b = z_seq[t]
                    for s in range(self.K):
                        w = gamma[t, s]
                        C_pair_all[s, a, b] += w
                        n_s[s] += w

            # --- M-step: π, Π ---
            self.pi = N0 + self.psi
            self.pi /= self.pi.sum()
            self.Pi = Nij + self.psi
            self.Pi /= self.Pi.sum(axis=1, keepdims=True)

            # --- M-step: η per state (gradient roll) ---
            for s in range(self.K):
                self._optimize_eta_for_state(s, C_pair_all[s], n_s[s], max_iter=200, lr=0.5)

            # --- M-step: Φ (weighted gaussian MLE through (s,i,a)) ---
            self._update_phi(all_stats)

            if verbose:
                print(f"EM iter {it:3d}: total loglik = {total_ll:.6f}")
            if np.abs(total_ll - prev_ll) < tol:
                if verbose:
                    print("Converged.")
                break
            prev_ll = total_ll
        return total_ll

    # ---------- генерация выборки ----------
    def sample(self, T: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = np.empty(T, dtype=int)
        z = np.empty((T,2), dtype=int)
        x = np.empty((T,2,self.F), dtype=float)

        s[0] = self.rng.choice(self.K, p=self.pi)
        for t in range(T):
            if t > 0:
                s[t] = self.rng.choice(self.K, p=self.Pi[s[t-1]])
            k = s[t]
            # sampliing (z1,z2) ~ softmax(scores)
            scores = self.theta1[k][:, None] + self.theta2[k][None, :] + self.W[k]
            probs = np.exp(scores - self.logsumexp(scores.ravel()))
            probs = probs / probs.sum()
            flat = probs.ravel()
            idx = self.rng.choice(self.A*self.A, p=flat)
            z1, z2 = idx // self.A, idx % self.A
            z[t] = [z1, z2]
            # sampling x_i ~ N(mean_{k,i,z_i}, diag(var_{k,i,z_i}))
            for i, zi in enumerate((z1, z2)):
                mean = self.phi_mean[k, i, zi]
                var = np.maximum(self.phi_var[k, i, zi], 1e-8)
                x[t, i] = self.rng.normal(loc=mean, scale=np.sqrt(var))
        return s, z, x

    # ---------- Viterbi ----------
    def viterbi(self, z_seq: np.ndarray, x_seq: np.ndarray) -> np.ndarray:
        """
        Находит MAP-путь состояний s_{1:T} по Viterbi в лог-домене.
        """
        T = z_seq.shape[0]
        logPi = np.log(np.maximum(self.Pi, self.EPS))
        log_delta = np.empty((T, self.K))
        psi = np.empty((T, self.K), dtype=int)

        # init
        z1, z2 = z_seq[0]
        x1, x2 = x_seq[0,0], x_seq[0,1]
        logB0 = np.array([self._log_emission(s, z1, z2, x1, x2) for s in range(self.K)])
        log_delta[0] = np.log(np.maximum(self.pi, self.EPS)) + logB0

        for t in range(1, T):
            z1, z2 = z_seq[t]
            x1, x2 = x_seq[t,0], x_seq[t,1]
            logB = np.array([self._log_emission(s, z1, z2, x1, x2) for s in range(self.K)])
            for j in range(self.K):
                vals = log_delta[t-1] + logPi[:, j]
                psi[t, j] = np.argmax(vals)
                log_delta[t, j] = np.max(vals) + logB[j]

        states = np.empty(T, dtype=int)
        states[-1] = np.argmax(log_delta[-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states
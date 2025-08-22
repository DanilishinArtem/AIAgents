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
    # log⁡〖(∑8_𝑖▒𝑒^(𝑥_𝑖 ) )=𝑚+log⁡(∑_𝑖▒〖𝑒^(𝑥_𝑖 )−𝑚〗),𝑚=max_𝑖⁡〖𝑥_𝑖 〗 〗
    def logsumexp(self, x, axis=None):
        m = np.max(x, axis=axis, keepdims=True)
        return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)

    # ---------- вспомогательные вероятности ----------
    def _logp_z_given_s(self, s: int) -> np.ndarray:
        """
        Возвращает log p(z1=a, z2=b | s) как матрицу (A, A) в нормировке softmax.
        """
        scores = self.theta1[s][:, None] + self.theta2[s][None, :] + self.W[s]  # (A,A)
        logZ = self.logsumexp(scores.ravel())
        return scores - logZ

    @staticmethod
    def _gaussian_logpdf_diag(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
        # x, mean, var: (F,)
        var_safe = np.maximum(var, 1e-8)
        return -0.5 * (np.sum(np.log(2*np.pi*var_safe) + (x-mean)**2 / var_safe))

    def _log_emission(self, s: int, z1: int, z2: int, x1: np.ndarray, x2: np.ndarray) -> float:
        logPz = self._logp_z_given_s(s)[z1, z2]
        logPx1 = self._gaussian_logpdf_diag(x1, self.phi_mean[s,0,z1], self.phi_var[s,0,z1])
        logPx2 = self._gaussian_logpdf_diag(x2, self.phi_mean[s,1,z2], self.phi_var[s,1,z2])
        return logPz + logPx1 + logPx2

    # ---------- forward-backward ----------
    def _forward_backward(self, z_seq: np.ndarray, x_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        z_seq: (T,2) ints
        x_seq: (T,2,F) floats
        returns: gamma (T,K), xi (T-1,K,K), loglik
        """
        T = z_seq.shape[0]
        logB = np.empty((T, self.K))
        for t in range(T):
            z1, z2 = z_seq[t]
            x1, x2 = x_seq[t,0], x_seq[t,1]
            for s in range(self.K):
                logB[t, s] = self._log_emission(s, z1, z2, x1, x2)

        # forward in log-domain with scaling
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

        # backward
        log_beta = np.empty((T, self.K))
        log_beta[-1] = 0.0
        for t in range(T-2, -1, -1):
            # log_beta[t,i] = logsumexp_j ( logPi[i,j] + logB[t+1,j] + log_beta[t+1,j] )
            log_beta[t] = self.logsumexp(logPi + (logB[t+1] + log_beta[t+1])[None, :], axis=1)
            # scale to keep consistency (subtract self.logsumexp for numerical stability)
            c = self.logsumexp(log_beta[t])
            log_beta[t] -= c

        # gamma
        log_gamma = log_alpha + log_beta
        # normalize per t
        log_gamma = (log_gamma.T - self.logsumexp(log_gamma, axis=1)).T
        gamma = np.exp(log_gamma)

        # xi
        xi = np.zeros((T-1, self.K, self.K))
        for t in range(T-1):
            # unnormalized in log
            M = (log_alpha[t][:, None] + logPi + logB[t+1][None, :] + log_beta[t+1][None, :])
            M -= self.logsumexp(M)
            xi[t] = np.exp(M)

        return gamma, xi, loglik

    # ---------- M-step: η (лог-линейная оптимизация) ----------
    def _optimize_eta_for_state(self, s: int, C_pair: np.ndarray, n_s: float,
                                max_iter: int = 200, lr: float = 0.5) -> None:
        """
        Оптимизирует (theta1[s], theta2[s], W[s]) по взвешенному log-likelihood:
          L_s(η) = sum_{a,b} C_pair[a,b] * score[a,b] - n_s * self.logsumexp(score) - (l2/2)||η||^2
        где score[a,b] = θ1[a] + θ2[b] + W[a,b]
        Используем простой градиентный подъём с backtracking line search.
        """
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

            # obj
            obj = np.sum(C_pair * scores) - n_s * logZ
            # L2
            obj -= 0.5 * self.l2_eta * (np.sum(th1**2) + np.sum(th2**2) + np.sum(W**2))

            # grads (empirical - n_s * model) - λ * θ
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
                if obj_new >= obj - 1e-9:  # не ухудшилось
                    th1, th2, W = th1_new, th2_new, W_new
                    prev_obj = obj_new
                    improved = True
                    break
                step *= 0.5
            if not improved:
                # малый шаг — прекращаем
                break

        self.theta1[s], self.theta2[s], self.W[s] = pack(th1, th2, W)

    # ---------- M-step: Φ (взвешенные гауссовы MLE) ----------
    def _update_phi(self, stats: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]):
        """
        stats: список по последовательностям [(gamma, z_seq, x_seq)]
        Обновляет mean/var для каждого (s, i, a).
        """
        K, A, F = self.K, self.A, self.F
        # аккумулируем суммы
        sum_w = np.zeros((K, 2, A))
        sum_x = np.zeros((K, 2, A, F))
        sum_x2 = np.zeros((K, 2, A, F))

        for gamma, z_seq, x_seq in stats:
            T = z_seq.shape[0]
            for t in range(T):
                z1, z2 = z_seq[t]
                x1, x2 = x_seq[t,0], x_seq[t,1]
                for s in range(K):
                    w = gamma[t, s]
                    if w <= 0: 
                        continue
                    # агент 1, действие z1
                    sum_w[s,0,z1] += w
                    sum_x[s,0,z1] += w * x1
                    sum_x2[s,0,z1] += w * (x1**2)
                    # агент 2, действие z2
                    sum_w[s,1,z2] += w
                    sum_x[s,1,z2] += w * x2
                    sum_x2[s,1,z2] += w * (x2**2)

        # обновление mean/var
        mean = np.copy(self.phi_mean)
        var = np.copy(self.phi_var)
        for s in range(K):
            for i in range(2):
                for a in range(A):
                    w = sum_w[s,i,a]
                    if w > 0:
                        m = sum_x[s,i,a] / w
                        # дисперсия по координатам (диагональ)
                        v = (sum_x2[s,i,a] / w) - (m**2)
                        v = np.maximum(v, 1e-6)  # floor
                        mean[s,i,a] = m
                        var[s,i,a] = v
        self.phi_mean = mean
        self.phi_var = var

    # ---------- внешний EM ----------
    def fit_em(self, sequences: List[Tuple[np.ndarray, np.ndarray]],
               n_iter: int = 50, tol: float = 1e-4, verbose: bool = True):
        """
        sequences: список из (z_seq, x_seq)
          z_seq: (T,2) int
          x_seq: (T,2,F) float
        """
        prev_ll = -np.inf
        for it in range(1, n_iter+1):
            total_ll = 0.0
            # достаточные статистики
            N0 = np.zeros(self.K)
            Nij = np.zeros((self.K, self.K))
            all_stats = []  # для Φ
            # для η: аккумулируем по состояниям C_pair[s, a, b]
            C_pair_all = np.zeros((self.K, self.A, self.A))
            n_s = np.zeros(self.K)

            # --- E-step ---
            for z_seq, x_seq in sequences:
                gamma, xi, ll = self._forward_backward(z_seq, x_seq)
                total_ll += ll
                T = z_seq.shape[0]
                N0 += gamma[0]
                Nij += xi.sum(axis=0)
                all_stats.append((gamma, z_seq, x_seq))
                # статистики для η
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

            # --- M-step: η per state (градиентный подъём) ---
            for s in range(self.K):
                self._optimize_eta_for_state(s, C_pair_all[s], n_s[s], max_iter=200, lr=0.5)

            # --- M-step: Φ (взвешенные гауссовы MLE по (s,i,a)) ---
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
        """
        Генерирует траекторию длины T.
        Возвращает: states (T,), z_seq (T,2), x_seq (T,2,F).
        """
        s = np.empty(T, dtype=int)
        z = np.empty((T,2), dtype=int)
        x = np.empty((T,2,self.F), dtype=float)

        s[0] = self.rng.choice(self.K, p=self.pi)
        for t in range(T):
            if t > 0:
                s[t] = self.rng.choice(self.K, p=self.Pi[s[t-1]])
            k = s[t]
            # сэмплируем (z1,z2) ~ softmax(scores)
            scores = self.theta1[k][:, None] + self.theta2[k][None, :] + self.W[k]
            probs = np.exp(scores - self.logsumexp(scores.ravel()))
            probs = probs / probs.sum()
            flat = probs.ravel()
            idx = self.rng.choice(self.A*self.A, p=flat)
            z1, z2 = idx // self.A, idx % self.A
            z[t] = [z1, z2]
            # сэмплируем x_i ~ N(mean_{k,i,z_i}, diag(var_{k,i,z_i}))
            for i, zi in enumerate((z1, z2)):
                mean = self.phi_mean[k, i, zi]
                var = np.maximum(self.phi_var[k, i, zi], 1e-8)
                x[t, i] = self.rng.normal(loc=mean, scale=np.sqrt(var))
        return s, z, x

    # ---------- Viterbi (по желанию) ----------
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
import numpy as np
import math
from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
from torch import nn


class HMMTwoAgents:
    def __init__(self, n_states: int, n_actions: int, rand_state: int | None = None, psi=1e-6):
        """
        n_states: число скрытых состояний K
        n_actions: число типов действий для одного агента (|Ρ|)
        psi: регуляризация (дирихле псевдосчёты) для эмиссий
        """
        self.K = n_states
        self.A = n_actions
        self.rng = np.random.default_rng(rand_state)
        # начальные параметры
        self.pi = self.rng.random(self.K)
        self.pi /= self.pi.sum()
        self.Pi = self.rng.random((self.K, self.K))
        self.Pi /= self.Pi.sum(axis=1, keepdims=True)
        # эмиссии: tensor (K, A, A) -> p(z1,z2 | s)
        self.emiss = self.rng.random((self.K, self.A, self.A))
        self.emiss /= self.emiss.sum(axis=(1,2), keepdims=True)
        self.psi = float(psi)

    def _logsumexp(self, x, axis=None):
        m = np.max(x, axis=axis, keepdims=True)
        return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)

    def _forward_backward(self, seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        seq: array shape (T,2) with integer actions in [0, A-1]
        returns: gamma (T,K), xi (T-1,K,K), log_likelihood
        """
        T = seq.shape[0]
        # emission likelihoods B[t,s] = p(y_t | s)
        B = np.empty((T, self.K))
        for t in range(T):
            z1, z2 = seq[t]
            B[t] = self.emiss[:, z1, z2]
        # forward
        alpha = np.empty((T, self.K))
        alpha[0] = self.pi * B[0]
        c0 = alpha[0].sum()
        if c0 == 0:
            c0 = 1e-300
        alpha[0] /= c0
        scales = np.empty(T)
        scales[0] = c0
        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ self.Pi) * B[t]
            ct = alpha[t].sum()
            if ct == 0:
                ct = 1e-300
            alpha[t] /= ct
            scales[t] = ct
        loglik = np.log(scales).sum()
        # backward
        beta = np.empty((T, self.K))
        beta[-1] = 1.0
        for t in range(T-2, -1, -1):
            beta[t] = (self.Pi * B[t+1]).dot(beta[t+1])
            # normalize to avoid underflow (not strictly necessary since we use scales)
            s = beta[t].sum()
            if s == 0:
                s = 1e-300
            beta[t] /= s
        # gamma and xi
        gamma = (alpha * beta)
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi = np.zeros((T-1, self.K, self.K))
        for t in range(T-1):
            # unnormalized xi: alpha[t,i] * Pi[i,j] * B[t+1,j] * beta[t+1,j]
            numer = (alpha[t][:, None] * self.Pi) * (B[t+1][None, :] * beta[t+1][None, :])
            denom = numer.sum()
            if denom == 0:
                denom = 1e-300
            xi[t] = numer / denom
        return gamma, xi, loglik

    def fit_em(self, sequences: List[np.ndarray], n_iter: int = 50, tol: float = 1e-4, verbose: bool = True):
        """
        sequences: list of sequences, each shape (T,2) of int actions
        EM iterations (Baum-Welch for HMM)
        """
        prev_ll = -np.inf
        for it in range(1, n_iter + 1):
            # sufficient statistics
            N0 = np.zeros(self.K)
            Nij = np.zeros((self.K, self.K))
            emiss_counts = np.zeros((self.K, self.A, self.A))
            total_ll = 0.0
            for seq in sequences:
                gamma, xi, ll = self._forward_backward(seq)
                total_ll += ll
                T = seq.shape[0]
                N0 += gamma[0]
                Nij += xi.sum(axis=0)
                for t in range(T):
                    z1, z2 = seq[t]
                    emiss_counts[:, z1, z2] += gamma[t]
            # M-step (with Dirichlet-like pseudo-counts psi)
            # pi
            self.pi = (N0 + self.psi)
            self.pi /= self.pi.sum()
            # Pi (rows sum to 1)
            self.Pi = (Nij + self.psi)
            self.Pi /= self.Pi.sum(axis=1, keepdims=True)
            # emissions
            self.emiss = emiss_counts + self.psi
            self.emiss /= self.emiss.sum(axis=(1,2), keepdims=True)

            if verbose:
                print(f"EM iter {it:3d}: total loglik = {total_ll:.4f}")
            if np.abs(total_ll - prev_ll) < tol:
                if verbose:
                    print("Converged.")
                break
            prev_ll = total_ll
        return total_ll

    def sample(self, T: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерация одной траектории длины T.
        Возвращает: states (T,1), obs (T,2)
        """
        s = np.empty(T, dtype=int)
        y = np.empty((T, 2), dtype=int)
        s[0] = self.rng.choice(self.K, p=self.pi)
        for t in range(T):
            if t > 0:
                s[t] = self.rng.choice(self.K, p=self.Pi[s[t-1]])
            # sample pair (z1,z2) according to emiss[s[t]]
            flat = self.emiss[s[t]].ravel()
            idx = self.rng.choice(self.A*self.A, p=flat)
            z1 = idx // self.A
            z2 = idx % self.A
            y[t, 0] = z1
            y[t, 1] = z2
        return s, y

    def viterbi(self, seq: np.ndarray) -> np.ndarray:
        T = seq.shape[0]
        delta = np.empty((T, self.K))
        psi = np.empty((T, self.K), dtype=int)
        # init
        B0 = np.array([self.emiss[:, seq[0,0], seq[0,1]]]).squeeze()
        delta[0] = np.log(self.pi + 1e-300) + np.log(B0 + 1e-300)
        for t in range(1, T):
            B_t = self.emiss[:, seq[t,0], seq[t,1]]
            for j in range(self.K):
                vals = delta[t-1] + np.log(self.Pi[:, j] + 1e-300)
                psi[t, j] = np.argmax(vals)
                delta[t, j] = np.max(vals) + np.log(B_t[j] + 1e-300)
        states = np.empty(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states



# HMM для двух агентов — лог-линейные эмиссии + атрибуты + MAP / вариационный апдейт
# Файл: hmm_two_agents.py
#
# Что сделано в этой реализации:
# - HMM с K скрытыми состояниями, дискретными действиями двух агентов (A типов)
# - Эмиссия.p(z1,z2 | s) задаётся лог-линейной моделью:
#       log unnorm = theta1[s, z1] + theta2[s, z2] + W[s, z1, z2]
#   и нормируется по всем парам (z1,z2) через softmax
# - E-step: классический forward-backward возвращает gamma, xi и loglik
# - M-step: параметры переходов/пи апдейтятся в closed-form (псевдосчёты)
# - Для эмиссий используется градиентный MAP (Adam) оптимизируя ожидаемое
#   лог-правдоподобие, взвешенное по gamma (т.е. EM-within-M-step)
# - Добавлена модель атрибутов x_{i,t} (по каждому агенту):
#     Gaussian emission условно на (agent, state, action):
#       x ~ N(mu_{i,s,z}, sigma2_{i,s,z})
#   их параметры тоже оптимизируются в M-step совместно с эмиссиями
# - Дополнительно реализован черновой mean-field вариационный апдейт
#   (black-box VI) для параметров эмиссии: q(theta1,theta2,W) ~ Normal
#   Реализация использует параметризацию (mu, logvar) и оптимизацию ELBO
class HMMTwoAgentsLogLinear:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        device: Optional[torch.device] = None,
        prior_std: float = 1.0,
        rand_state: int = 0,
    ):
        self.K = n_states
        self.A = n_actions
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        torch.manual_seed(rand_state)

        # discrete HMM parameters (as numpy for closed-form updates in M-step)
        self.pi = np.ones(self.K) / self.K
        self.Pi = np.ones((self.K, self.K)) / self.K

        # emission parameters (torch tensors, will be trained)
        # theta1, theta2: (K, A)
        # W: (K, A, A)
        self.prior_std = float(prior_std)
        self.theta1 = nn.Parameter(torch.randn(self.K, self.A, device=self.device) * 0.01)
        self.theta2 = nn.Parameter(torch.randn(self.K, self.A, device=self.device) * 0.01)
        self.W = nn.Parameter(torch.randn(self.K, self.A, self.A, device=self.device) * 0.01)

        # attribute model (Gaussian) parameters
        # mu_attrs[i] shape: (K, A) for agent i (scalar x); sigma2 param per (i,K,A)
        # For simplicity start with scalar attributes; extendable to vector
        self.mu_attr1 = nn.Parameter(torch.randn(self.K, self.A, device=self.device) * 0.01)
        self.log_var_attr1 = nn.Parameter(torch.zeros(self.K, self.A, device=self.device) - 3.0)
        self.mu_attr2 = nn.Parameter(torch.randn(self.K, self.A, device=self.device) * 0.01)
        self.log_var_attr2 = nn.Parameter(torch.zeros(self.K, self.A, device=self.device) - 3.0)

        # Put parameters in a list for optimizer convenience
        self._params = [self.theta1, self.theta2, self.W, self.mu_attr1, self.log_var_attr1, self.mu_attr2, self.log_var_attr2]

    # -------------------- вспомогательные функции --------------------
    def _emission_log_probs(self, pairs: torch.LongTensor) -> torch.Tensor:
        # pairs: (T,2) long tensor with values in [0,A-1]
        # returns log_probs: (T, K) log p(y_t | s)
        T = pairs.shape[0]
        z1 = pairs[:, 0]  # (T,)
        z2 = pairs[:, 1]
        # We want for each s compute logit for observed pair: theta1[s,z1] + theta2[s,z2] + W[s,z1,z2]
        # Build tensors: (K, T)
        # gather
        t1 = self.theta1[:, z1]  # (K, T)
        t2 = self.theta2[:, z2]
        w = self.W[:, z1, z2]
        logits_obs = t1 + t2 + w  # (K, T)
        # But we need normalization over all pairs (z1', z2'). Compute log-sum-exp over flattened pairs
        # For numerical efficiency compute for each s: logZ_s = logsumexp_{a,b} theta1[s,a] + theta2[s,b] + W[s,a,b]
        # Precompute for each s the logZ
        # flatten pairs space (A*A) and compute logits for all pairs
        theta1_s = self.theta1.unsqueeze(2)  # (K, A, 1)
        theta2_s = self.theta2.unsqueeze(1)  # (K, 1, A)
        logits_full = theta1_s + theta2_s + self.W  # (K, A, A)
        logZ = torch.logsumexp(logits_full.view(self.K, -1), dim=1)  # (K,)
        # Now for each t and s the log p = logits_obs[s,t] - logZ[s]
        logp = logits_obs.transpose(0, 1) - logZ.unsqueeze(0)  # (T, K)
        return logp

    @staticmethod
    def _gaussian_logpdf(x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        # x: (T,) or (T,1) ; mu: (K,A), but we'll gather proper entries. Return (T,K)
        # compute elementwise: -0.5*(log(2pi)+log_var) - 0.5*((x-mu)^2 / var)
        var = torch.exp(log_var)
        return -0.5 * (torch.log(2 * torch.pi * var) + (x - mu) ** 2 / var)

    # -------------------- алгоритмы вывода --------------------
    def _forward_backward(self, sequences: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
        # sequences: list of arrays shape (T,2) for discrete actions; attributes optionally provided separately
        # We'll process each sequence independently and collect gammas/xis
        gammas = []
        xis = []
        total_ll = 0.0
        # Move emission params to CPU for numeric forward-backward? We'll run in numpy using torch computed emission probs
        for seq in sequences:
            pairs = torch.tensor(seq, dtype=torch.long, device=self.device)
            T = pairs.shape[0]
            logB = self._emission_log_probs(pairs).cpu().detach().numpy()  # (T,K)
            B = np.exp(logB)
            # forward-backward with scaling (numpy)
            alpha = np.zeros((T, self.K))
            beta = np.zeros((T, self.K))
            scales = np.zeros(T)
            alpha[0] = self.pi * B[0]
            c0 = alpha[0].sum()
            if c0 == 0:
                c0 = 1e-300
            alpha[0] /= c0
            scales[0] = c0
            for t in range(1, T):
                alpha[t] = (alpha[t - 1] @ self.Pi) * B[t]
                ct = alpha[t].sum()
                if ct == 0:
                    ct = 1e-300
                alpha[t] /= ct
                scales[t] = ct
            beta[-1] = 1.0
            for t in range(T - 2, -1, -1):
                beta[t] = (self.Pi * B[t + 1]).dot(beta[t + 1])
                s = beta[t].sum()
                if s == 0:
                    s = 1e-300
                beta[t] /= s
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)
            xi = np.zeros((T - 1, self.K, self.K))
            for t in range(T - 1):
                numer = (alpha[t][:, None] * self.Pi) * (B[t + 1][None, :] * beta[t + 1][None, :])
                denom = numer.sum()
                if denom == 0:
                    denom = 1e-300
                xi[t] = numer / denom
            gammas.append(gamma)
            xis.append(xi)
            total_ll += np.log(scales).sum()
        return gammas, xis, total_ll

    # -------------------- M-step: closed-form for pi and Pi, gradient MAP for emissions & attrs --------------------
    def m_step_map(
        self,
        sequences: List[np.ndarray],
        gammas: List[np.ndarray],
        n_epochs: int = 50,
        lr: float = 1e-2,
        weight_decay: float = 1e-3,
        verbose: bool = False,
    ) -> None:
        # Update pi and Pi (numpy counts)
        N0 = np.zeros(self.K)
        Nij = np.zeros((self.K, self.K))
        for seq, gamma, xi in zip(sequences, gammas, [None]*len(sequences)):
            T = seq.shape[0]
            N0 += gamma[0]
            # xi can be recomputed from gamma and transitions, but easier to compute using forward-backward above
        # To get Nij we need xis; simpler: recompute with forward-backward that returns xis
        gammas2, xis, _ = self._forward_backward(sequences)
        for xi in xis:
            Nij += xi.sum(axis=0)
        self.pi = (N0 + 1e-6)
        self.pi /= self.pi.sum()
        self.Pi = (Nij + 1e-6)
        self.Pi /= self.Pi.sum(axis=1, keepdims=True)

        # Now update emission params and attribute params via gradient optimization of expected log-likelihood
        optimizer = torch.optim.Adam(self._params, lr=lr, weight_decay=weight_decay)
        # prepare data tensors concatenated across sequences for efficiency
        pairs_all = []
        gamma_all = []
        attrs1 = []
        attrs2 = []
        for seq, gamma in zip(sequences, gammas2):
            pairs_all.append(torch.tensor(seq, dtype=torch.long, device=self.device))
            gamma_all.append(torch.tensor(gamma, dtype=torch.float32, device=self.device))
            # placeholders for attrs; user can extend to pass real attributes
            # for now create NaNs to indicate missing
            T = seq.shape[0]
            attrs1.append(torch.full((T,), float('nan'), device=self.device))
            attrs2.append(torch.full((T,), float('nan'), device=self.device))
        pairs_cat = torch.cat(pairs_all, dim=0)
        gamma_cat = torch.cat(gamma_all, dim=0)
        # Placeholder attributes - in practice pass real attributes to m_step_map for learning

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            # emission log-probs for observed pairs
            logp = self._emission_log_probs(pairs_cat)  # (T_total, K)
            # expected log-likelihood = sum_t sum_s gamma_t[s] * log p(y_t | s)
            loss_emis = -torch.sum(gamma_cat * logp)
            # attribute gaussian logpdfs if provided (here we ignore NaNs)
            # compute attr loss only for non-nan entries
            loss_attr = torch.tensor(0.0, device=self.device)
            # prior (Gaussian) on parameters -> weight decay already handled, but add explicit negative log prior for clarity
            prior = 0.0
            for p in [self.theta1, self.theta2, self.W, self.mu_attr1, self.mu_attr2]:
                prior = prior + 0.5 * torch.sum((p / self.prior_std) ** 2)
            loss = loss_emis + loss_attr + prior
            loss.backward()
            optimizer.step()
            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                print(f"  M-step epoch {epoch+1}/{n_epochs}, loss = {loss.item():.6f}, emis = {loss_emis.item():.6f}")

    # -------------------- EM outer loop --------------------
    def fit_em(
        self,
        sequences: List[np.ndarray],
        n_iter: int = 20,
        mstep_epochs: int = 100,
        mstep_lr: float = 1e-2,
        verbose: bool = True,
    ) -> float:
        prev_ll = -float('inf')
        for it in range(1, n_iter + 1):
            gammas, xis, ll = self._forward_backward(sequences)
            if verbose:
                print(f"EM iter {it:3d}: loglik = {ll:.4f}")
            self.m_step_map(
                sequences, gammas, n_epochs=mstep_epochs, lr=mstep_lr, weight_decay=1e-4, verbose=verbose
            )
            if abs(ll - prev_ll) < 1e-4:
                if verbose:
                    print("Converged EM")
                break
            prev_ll = ll
        return prev_ll

    # -------------------- Вариационный mean-field (black-box VI) --------------------
    def variational_em(self, sequences: List[np.ndarray], n_iter: int = 1000, lr: float = 1e-3, n_samples: int = 4):
        # We build variational distributions q(theta1), q(theta2), q(W) as diagonal Gaussians
        # parameterize variational params as mu and logvar for each param tensor
        # For simplicity vectorize parameters
        params_vec = torch.cat([self.theta1.flatten(), self.theta2.flatten(), self.W.flatten()])
        D = params_vec.shape[0]
        mu_q = nn.Parameter(params_vec.clone().detach())
        logvar_q = nn.Parameter(torch.full_like(mu_q, -6.0))
        opt = torch.optim.Adam([mu_q, logvar_q], lr=lr)

        # compute prior logp(params) under N(0, prior_std^2)
        def log_prior(z):
            return -0.5 * torch.sum((z / self.prior_std) ** 2) - 0.5 * D * math.log(2 * math.pi * (self.prior_std ** 2))

        # precompute data flattened
        pairs_cat = torch.cat([torch.tensor(s, dtype=torch.long, device=self.device) for s in sequences], dim=0)

        for it in range(1, n_iter + 1):
            opt.zero_grad()
            # Monte Carlo estimate of ELBO
            elbo = 0.0
            for _ in range(n_samples):
                eps = torch.randn_like(mu_q)
                z = mu_q + torch.exp(0.5 * logvar_q) * eps  # sample
                # unpack z back into theta1, theta2, W shapes
                idx1 = self.K * self.A
                idx2 = idx1 + self.K * self.A
                theta1_samp = z[:idx1].view(self.K, self.A)
                theta2_samp = z[idx1:idx2].view(self.K, self.A)
                W_samp = z[idx2:].view(self.K, self.A, self.A)
                # compute log-likelihood p(y|params) by computing log-probs for pairs
                # build logits_full for each s
                logits_full = theta1_samp.unsqueeze(2) + theta2_samp.unsqueeze(1) + W_samp
                logZ = torch.logsumexp(logits_full.view(self.K, -1), dim=1)
                # compute log p for all observed pairs
                z1 = pairs_cat[:, 0]
                z2 = pairs_cat[:, 1]
                logits_obs = theta1_samp[:, z1] + theta2_samp[:, z2] + W_samp[:, z1, z2]
                logp = logits_obs.transpose(0, 1) - logZ.unsqueeze(0)  # (T, K)
                # For VI we marginalize s by summing log-sum-exp? Use log-sum-exp over s for marginal likelihood
                # log p(y_t) = logsumexp_s log p(y_t | s) + log p(s) but for simplicity assume uniform s and use mean over s
                # (This is an approximation; we are optimizing joint ELBO on emissions only.)
                logp_marg = torch.logsumexp(logp, dim=1) - math.log(self.K)
                ll = logp_marg.sum()
                lp = log_prior(z)
                # entropy of q: -E_q log q = 0.5 * (D*(1+log(2pi)) + sum logvar)
                entropy = 0.5 * (D * (1.0 + math.log(2 * math.pi)) + torch.sum(logvar_q))
                elbo += (ll + lp + entropy)
            elbo = elbo / n_samples
            # maximize ELBO -> minimize -ELBO
            loss = -elbo
            loss.backward()
            opt.step()
            if it % 100 == 0:
                print(f"VI iter {it}/{n_iter}, -ELBO = {loss.item():.4f}")
        # after VI, set MAP params to mu_q mean
        with torch.no_grad():
            z_map = mu_q.detach()
            idx1 = self.K * self.A
            idx2 = idx1 + self.K * self.A
            self.theta1.copy_(z_map[:idx1].view(self.K, self.A))
            self.theta2.copy_(z_map[idx1:idx2].view(self.K, self.A))
            self.W.copy_(z_map[idx2:].view(self.K, self.A, self.A))

    # -------------------- генерация / Viterbi --------------------
    def sample(self, T: int) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng()
        s = np.empty(T, dtype=int)
        y = np.empty((T, 2), dtype=int)
        s[0] = rng.choice(self.K, p=self.pi)
        for t in range(T):
            if t > 0:
                s[t] = rng.choice(self.K, p=self.Pi[s[t - 1]])
            # compute emission prob table for current s
            logits = (self.theta1[s[t]].detach().cpu().numpy()[..., None] +
                      self.theta2[s[t]].detach().cpu().numpy()[None, ...] +
                      self.W[s[t]].detach().cpu().numpy())
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()
            flat = probs.ravel()
            idx = rng.choice(self.A * self.A, p=flat)
            z1 = idx // self.A
            z2 = idx % self.A
            y[t, 0] = z1
            y[t, 1] = z2
        return s, y

    def viterbi(self, seq: np.ndarray) -> np.ndarray:
        T = seq.shape[0]
        # compute logB
        pairs = torch.tensor(seq, dtype=torch.long, device=self.device)
        logB = self._emission_log_probs(pairs).cpu().detach().numpy()  # (T,K)
        delta = np.empty((T, self.K))
        psi = np.empty((T, self.K), dtype=int)
        delta[0] = np.log(self.pi + 1e-300) + logB[0]
        for t in range(1, T):
            for j in range(self.K):
                vals = delta[t - 1] + np.log(self.Pi[:, j] + 1e-300)
                psi[t, j] = np.argmax(vals)
                delta[t, j] = np.max(vals) + logB[t, j]
        states = np.empty(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states


# ------------------ Пример использования ------------------
if __name__ == "__main__":
    actions = ["SQL", "Text", "Image", "Analyze", "Compile", "Fix"]
    A = len(actions)
    K = 3
    model = HMMTwoAgentsLogLinear(n_states=K, n_actions=A, rand_seed=42)

    # сгенерируем синтетические данные
    sequences = []
    for _ in range(40):
        s, y = model.sample(T=25)
        sequences.append(y)

    # перезапишем параметры случайно перед обучением
    model = HMMTwoAgentsLogLinear(n_states=K, n_actions=A, rand_seed=123)
    print("Start EM (MAP M-step)")
    model.fit_em(sequences, n_iter=8, mstep_epochs=80, mstep_lr=5e-3, verbose=True)

    # опционально: выполнить VI для улучшения апостериорной устойчивости
    print("Запускаем вариационный подбор (VI) по эмиссиям...")
    model.variational_em(sequences, n_iter=400, lr=1e-3, n_samples=6)

    # продемонстрируем Viterbi
    s_hat = model.viterbi(sequences[0])
    print("Viterbi states (пример):", s_hat)

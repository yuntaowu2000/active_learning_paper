import gc
import os

import torch

from deep_macrofin import (LossReductionMethod, OptimizerType, PDEModel,
                           PDEModelTimeStep, SamplingMethod, set_seeds)

from common import *
from equilibrium import compute_sv_equilibrium

# ===========================================================================
# PDEModel subclass: simplex+v sampler, RAR override, fused forward.
# ===========================================================================
class _SVNAgentMixin:
    """Shared logic for the stationary and time-stepping N-agent SV models."""

    def _sv_init(self, config):
        self.rar = config.get("rar", False)
        # Dirichlet-alpha mixture range for wealth-share sampling (see
        # _mixture_shares_torch); configurable so the concentration can be swept.
        self.share_alpha_lo = config.get("share_alpha_lo", SHARE_ALPHA_LO)
        self.share_alpha_hi = config.get("share_alpha_hi", SHARE_ALPHA_HI)
        self.statics = None
        # Names of the stacked networks (recorded by ``attach_stacks``). The
        # actual batched forward is done by the library's stacked evaluator; we
        # only keep the names so ``update_variables`` can re-assemble the stacked
        # tensors from the per-network keys it writes into the value dict.
        self._xi_names = []
        self._p_name = "p"
        self._r_name = "r"
        # enable the library's experimental vmap-batched local-function evaluation
        self.stacked = True
        # disable the per-epoch diagnostic (expensive, unused here)
        try:
            self._PDEModel__compute_changes = lambda SV: {"total": 0.0}
        except Exception:
            pass

    # -- stack registration --------------------------------------------------
    def attach_stacks(self, xi_names, p_name="p", r_name="r"):
        # Record which networks feed the equilibrium.  We DON'T hold references to
        # the underlying modules here: the library's stacked evaluator is rebuilt
        # lazily (and invalidated automatically inside ``load_model``), so the
        # time-stepping outer loop -- which rebuilds every agent/endog module each
        # iteration -- always evaluates the CURRENT networks.  We just remember
        # the names.
        self._xi_names = list(xi_names)
        self._p_name = p_name
        self._r_name = r_name
        self._invalidate_stacked_evaluator()

    # -- expert capital allocation -----------------------------------------
    def _compute_theta_E(self, x_full):
        """Expert capital shares ``theta_E`` (B, n_E), ordered as ``expert_idx``.

        Interior FOC: every expert is at its optimum,
        ``theta_k/x_k = chi/(gamma_k (phi v)^2)``; imposing capital clearing
        (``sum_k theta_k = 1``) pins ``chi`` analytically, so
        ``theta_k = (x_k/gamma_k) / sum_j(x_j/gamma_j)``.  This is a closed form
        of the state (leverage monotone in gamma by construction, capital-FOC
        residual identically 0), so NO theta networks are needed.
        """
        eidx = self.statics["expert_idx"]
        g_E = self.statics["gamma"][:, eidx]              # (1, n_E)
        w = x_full[:, eidx] / g_E                         # (B, n_E) = x_k/gamma_k
        return w / w.sum(dim=1, keepdim=True)

    # -- forward / equation evaluation --------------------------------------
    def update_variables(self, SV, vd=None):
        # This is the SINGLE place the forward is customized.  Everything else
        # (loss_fn, closure, validation, refinement scoring, outer-loop change
        # tracking) calls this method, so the equilibrium is computed uniformly.
        if vd is None:
            vd = self.variable_val_dict
        SV.requires_grad_(True)
        for i, sv_name in enumerate(self.state_variables):
            vd[sv_name] = SV[:, i:i+1]
        vd["SV"] = SV

        # Batched forward of every agent/endog network + its derivatives.  With
        # self.stacked == True this fuses the same-architecture networks (all the
        # xi_k and p) into one vmap call; it writes xi_k / xi_k_Jac / xi_k_Hess,
        # p / p_Jac / p_Hess and r into ``vd``.
        self._eval_local_functions(SV, vd)

        # Re-assemble the per-network slices into the stacked tensors the
        # equilibrium core expects.  vd[name] is (B, 1); vd[name_Jac] is
        # (B, 1, D); vd[name_Hess] is (B, 1, D, D), so concatenating over dim 1
        # across the K networks yields (B, K), (B, K, D), (B, K, D, D).
        xi = torch.cat([vd[n] for n in self._xi_names], dim=1)                 # (B, K)
        xi_Jac = torch.cat([vd[n + "_Jac"] for n in self._xi_names], dim=1)    # (B, K, D)
        xi_Hess = torch.cat([vd[n + "_Hess"] for n in self._xi_names], dim=1)  # (B, K, D, D)
        p = vd[self._p_name]                                                   # (B, 1)
        p_Jac = vd[self._p_name + "_Jac"]                                      # (B, 1, D)
        p_Hess = vd[self._p_name + "_Hess"]                                    # (B, 1, D, D)
        r = vd[self._r_name]                                                   # (B, 1) free rate

        # expert capital shares from the analytic interior FOC (no theta networks)
        K = self.statics["K"]
        x_states = SV[:, :K - 1]
        x_full = torch.cat([x_states, 1.0 - x_states.sum(dim=1, keepdim=True)], dim=1)
        theta_E = self._compute_theta_E(x_full)

        out = compute_sv_equilibrium(SV, xi, xi_Jac, xi_Hess, p, p_Jac, p_Hess, theta_E, r=r, statics=self.statics)

        vd["xi_active"] = xi
        vd.update(out)                             # includes r (== the free network here)

        for eq_name in self.equations:
            lhs = self.equations[eq_name].lhs.formula_str
            vd[lhs] = self.equations[eq_name].eval(self.custom_function_dict, vd)
    
    def closure(self, SV):
        for i, sv_name in enumerate(self.state_variables):
            self.variable_val_dict[sv_name] = SV[:, i:i+1]
        self.variable_val_dict["SV"] = SV
        self.update_variables(SV)
        self.loss_fn()
        total_loss = 0
        for loss_label, loss in self.loss_val_dict.items():
            total_loss += self.loss_weight_dict[loss_label] * torch.where(loss.isnan(), 0.0, loss)

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], max_norm=1.0)
        return total_loss

    def sample_simplex_v(self, epoch):
        """
        Dirichlet-alpha MIXTURE over the K wealth shares (drop residual) +
        uniform v.  The per-row log-uniform alpha spans egalitarian (alpha~1)
        and concentrated (alpha<1) wealth distributions -- essential at large K,
        where Dirichlet(1) would pin every share near 1/K (see
        _mixture_shares_torch).
        """
        K = self.statics["K"]
        eps = 0.1 / K
        shares = mixture_shares_torch(self.batch_size, K, eps, self.device, self.share_alpha_lo, self.share_alpha_hi)
        x_states = shares[:, :K - 1]                          # (B, K-1)
        vlo, vhi = self.statics["v_domain"]
        v = vlo + (vhi - vlo) * torch.rand((self.batch_size, 1), device=self.device)
        return torch.cat([x_states, v], dim=1)


class PDEModelNAgentsSV(_SVNAgentMixin, PDEModel):
    def __init__(self, name, config, latex_var_mapping={}):
        super().__init__(name, config, latex_var_mapping)
        self._sv_init(config)
        if self.rar:
            self.sample = self.sample_rar_greedy
            self.sampling_method = SamplingMethod.RARG
        else:
            self.sample = self.sample_simplex_v

    # NOTE: we deliberately do NOT override `closure`; the library default
    # (no gradient clipping) is used, matching the original 2-agent model.
    # Our `update_variables` override is invoked by that default closure.

    # -- residual-adaptive refinement ---------------------------------------
    def _refinement_loss_dict(self, epoch):
        self.set_all_model_eval()
        all_SVs, all_loss = [], []
        saved_bs = self.batch_size
        if self.statics["K"] > 20:
            self.batch_size = 500
            sample_times = 20
        else:
            self.batch_size = 1000
            sample_times = 10
        for _ in range(sample_times):
            torch.cuda.empty_cache()
            # values-only scoring -> run under no_grad so no autograd graph (incl.
            # the (B,N,D,D) Hessians) is retained.  torch.func still differentiates
            # the state-derivatives internally (see _score_pool note).
            with torch.no_grad():
                SV = self.sample_simplex_v(epoch)
                vd_ = self.variable_val_dict.copy()
                for i, sv_name in enumerate(self.state_variables):
                    vd_[sv_name] = SV[:, i:i + 1]
                vd_["SV"] = SV
                self.update_variables(SV, vd=vd_)
                total = torch.zeros((SV.shape[0], 1), device=self.device)
                Bn = SV.shape[0]

                def per_sample(res):
                    aa = torch.abs(res)
                    if aa.dim() == 0:
                        return aa.expand(Bn, 1).reshape(Bn, 1)
                    if aa.dim() == 1:
                        return aa.reshape(Bn, 1)
                    return aa.reshape(Bn, -1).mean(dim=-1, keepdim=True)

                for label in self.endog_equations:
                    total = total + per_sample(self.endog_equations[label].eval_no_loss(self.custom_function_dict, vd_))
                for label in self.hjb_equations:
                    total = total + per_sample(self.hjb_equations[label].eval_no_loss(self.custom_function_dict, vd_))
                all_SVs.append(SV.detach().cpu())
                all_loss.append(total.detach().cpu())
            del SV, total, vd_
            gc.collect(); torch.cuda.empty_cache()
        self.batch_size = saved_bs
        self.set_all_model_training()
        return {"SV": torch.cat(all_SVs, 0), "loss": torch.cat(all_loss, 0)}

    def sample_rar_greedy(self, epoch):
        if self.num_epochs and epoch % max(1, self.num_epochs // self.refinement_rounds) == 0 and epoch > 0:
            rd = self._refinement_loss_dict(epoch)
            ids = torch.topk(rd["loss"], self.batch_size // self.refinement_rounds, dim=0)[1].squeeze(-1)
            self.anchor_points = torch.vstack((self.anchor_points, rd["SV"][ids].to(self.device)))
        sv = self.sample_simplex_v(epoch)
        if self.anchor_points is not None and len(self.anchor_points) > 0:
            return torch.vstack((sv, self.anchor_points))
        return sv


class PDEModelTimeStepNAgentsSV(_SVNAgentMixin, PDEModelTimeStep):
    def __init__(self, name, config, latex_var_mapping={}):
        super().__init__(name, config, latex_var_mapping)
        self._sv_init(config)
        self.sample = self.sample_simplex_v_ts

        self.sample_boundary_cond = self.__sample_custom_boundary_cond
        self.boundary_uniform_points = None
        self._outer_iter = 0

        # --- learning-rate step decay along the backward (outer) march ---------
        # The library rebuilds the optimizer from self.lr at the START of every
        # outer loop and then fires OnInnerLoopStart once (with no args) right
        # after.  We hook that to (a) recompute self.lr from the base lr on a step
        # schedule and (b) patch the just-built optimizer's param groups, so the
        # decay takes effect on the current outer loop too.  Decay multiplies lr
        # by lr_decay_gamma every lr_decay_every outer iterations.
        self._lr_base = config.get("lr", self.lr)
        self.lr_decay_every = int(config.get("lr_decay_every", 20))
        self.lr_decay_gamma = float(config.get("lr_decay_gamma", 0.5))
        self._lr_decay_outer = 0
        self.OnInnerLoopStart += self._lr_decay_step

    def _lr_decay_step(self):
        """Step the LR every ``lr_decay_every`` outer iterations.  Fired once per
        outer loop (no args) via OnInnerLoopStart, so the call count equals the
        outer-iteration index.  Idempotent: lr is always recomputed from the base
        lr, so re-firing never compounds."""
        k = self._lr_decay_outer
        self._lr_decay_outer += 1
        if self.lr_decay_every and self.lr_decay_every > 0:
            factor = self.lr_decay_gamma ** (k // self.lr_decay_every)
        else:
            factor = 1.0
        new_lr = self._lr_base * factor
        self.lr = new_lr
        opt = getattr(self, "optimizer", None)
        if opt is not None:
            for pg in opt.param_groups:
                pg["lr"] = new_lr

    def sample_simplex_v_ts(self, epoch=0):
        """Simplex over shares + uniform v + uniform t in [min_t, max_t]."""
        base = self.sample_simplex_v(epoch)                   # (B, K) -> shares + v
        B = base.shape[0]
        min_t = self.config.get("min_t", 0.0)
        max_t = self.config.get("max_t", 1.0)
        t0_frac = float(self.config.get("t0_frac", 0.4))
        n_t0 = int(round(t0_frac * B))
        t = min_t + (max_t - min_t) * torch.rand((B, 1), device=self.device)
        t[:n_t0] = min_t
        return torch.cat([base, t], dim=1)
    
    def sample_simplex_v_random_t(self, epoch=0):
        """Wealth-simplex + v + one uniform-random ``t`` in ``[min_t, max_t]`` per
        point.  Used to score the interior ``(x, v, t)`` domain in
        ``sample_rar_greedy`` (distinct from the training sampler, which
        over-weights ``t=0``)."""
        base = self.sample_simplex_v(epoch)                   # (B, K) -> shares + v
        min_t = self.config.get("min_t", 0.0)
        max_t = self.config.get("max_t", 1.0)
        t = min_t + (max_t - min_t) * torch.rand((base.shape[0], 1), device=self.device)
        return torch.cat([base, t], dim=1)
    
    def __sample_custom_boundary_cond(self, time_val: float):
        if self.boundary_uniform_points is None:
            self.boundary_uniform_points = self.sample_simplex_v(0)
        time_dim = torch.ones((self.boundary_uniform_points.shape[0], 1), device=self.device) * time_val
        return torch.cat([self.boundary_uniform_points, time_dim], dim=-1)

    def _score_pool(self, sampler, rounds=5):
        """Sample ``rounds`` dense pools with ``sampler`` and return
        ``(SV_cpu, residual_cpu)`` where the residual is the per-point sum of
        |endog| + |hjb| equation residuals (the same equilibrium objects the
        training loss uses).  Heavy tensors are moved to CPU between pools to
        keep peak memory bounded."""
        SVs, losses = [], []
        for _ in range(rounds):
            # Scoring only needs residual VALUES for the topk ranking -- never a
            # backward pass.  The state-derivatives (xi_Jac/xi_Hess) come from
            # torch.func transforms, which differentiate independently of the
            # outer grad mode, so we run the whole forward under no_grad.  Without
            # this, an autograd graph (incl. the (B,N,D,D) Hessians) is built and
            # held for every endog/HJB residual -> the VRAM growth seen in RAR.
            with torch.no_grad():
                SV = sampler()
                vd_ = self.variable_val_dict.copy()
                for i, sv_name in enumerate(self.state_variables):
                    vd_[sv_name] = SV[:, i:i + 1]
                vd_["SV"] = SV
                self.update_variables(SV, vd=vd_)

                Bn = SV.shape[0]
                total = torch.zeros((Bn, 1), device=self.device)

                def _per_sample(res):
                    a = torch.abs(res)
                    return a.reshape(Bn, -1).mean(dim=-1, keepdim=True)

                for label in self.endog_equations:
                    total = total + _per_sample(self.endog_equations[label].eval_no_loss(self.custom_function_dict, vd_))
                for label in self.hjb_equations:
                    total = total + _per_sample(self.hjb_equations[label].eval_no_loss(self.custom_function_dict, vd_))

                SVs.append(SV.detach().cpu()); losses.append(total.detach().cpu())
            del SV, total, vd_
            gc.collect(); torch.cuda.empty_cache()
        return torch.cat(SVs, 0), torch.cat(losses, 0)

    def _sample_simplex_at_t0(self):
        """Wealth-simplex pool pinned to ``t = min_t`` -- the slice we actually
        extract as the stationary solution."""
        base = self.sample_simplex_v(0)
        min_t = self.config.get("min_t", 0.0)
        t = torch.full((base.shape[0], 1), min_t, device=self.device, dtype=base.dtype)
        return torch.cat([base, t], dim=1)

    def sample_rar_greedy(self):
        """Residual-based anchor accumulation, mirroring the library's
        ``PDEModelTimeStep.sample_rar_greedy`` (which calls
        ``__get_refinement_loss_dict`` + topk + vstack onto anchor_points).

        IMPORTANT contract (matches the library): this method takes no epoch
        argument and does NOT resample/return a training batch.  It only GROWS
        ``self.anchor_points`` with the highest-residual points from a dense
        pool.  The library inner loop decides *when* to call it (a few epochs
        per outer loop, gated on epoch>0) and then vstacks anchor_points onto
        the current SV batch itself.

        We use the stacked forward (``update_variables``) and the simplex+t
        sampler, because the equilibrium residual is not reachable through the
        library's local_function_dict path, and the domain is the wealth simplex
        (uniform sampling would be invalid).
        """
        self.set_all_model_eval()
        saved_bs = self.batch_size
        if self.statics["K"] > 20:
            self.batch_size = 500
            sample_times = 10
        else:
            self.batch_size = 1000
            sample_times = 5

        n_keep = min(max(2, self.batch_size // self.refinement_rounds), 5000)
        k_t0 = max(1, n_keep // 2)            # budget pinned to the t=min_t slice
        k_int = max(1, n_keep - k_t0)         # budget on the interior (x, t) domain

        SV_int, l_int = self._score_pool(self.sample_simplex_v_random_t, sample_times)
        SV_t0, l_t0 = self._score_pool(self._sample_simplex_at_t0, sample_times)

        self.batch_size = saved_bs
        self.set_all_model_training()
        
        ids_int = torch.topk(l_int, min(k_int, SV_int.shape[0]), dim=0)[1].squeeze(-1)
        ids_t0 = torch.topk(l_t0, min(k_t0, SV_t0.shape[0]), dim=0)[1].squeeze(-1)
        new_anchors = torch.vstack((SV_int[ids_int], SV_t0[ids_t0])).detach().to(self.device)

        if self.anchor_points is None or self.anchor_points.numel() == 0:
            self.anchor_points = new_anchors
        else:
            self.anchor_points = torch.vstack((self.anchor_points, new_anchors))
        del SV_int, l_int, SV_t0, l_t0
        gc.collect(); torch.cuda.empty_cache()
        return new_anchors

    # -- outer-loop convergence / variable tracking -------------------------
    # The library's __check_outer_loop_converge rebuilds every tracked variable
    # from self.local_function_dict + self.equations.  Our equilibrium (chat, r,
    # mu_P, ...) is produced by the custom `update_variables` forward instead --
    # and r is now ANALYTIC (no network, no equation), so the library version
    # would KeyError on it.  We override (matching the mangled name so the call
    # at train_model resolves here) to run our forward, then measure the change
    # against prev_vals exactly as the base class does.
    def _PDEModelTimeStep__check_outer_loop_converge(self, SV_T0):
        temp_dict = {}
        self.update_variables(SV_T0, vd=temp_dict)

        new_vals = {k: temp_dict[k].detach() for k in self.prev_vals}

        max_abs_change = 0.0
        max_rel_change = 0.0
        all_changes = {}
        for k in self.prev_vals:
            mean_new_val = torch.mean(new_vals[k]).item()
            abs_change = torch.mean(torch.abs(new_vals[k] - self.prev_vals[k])).item()
            rel_change = torch.mean(torch.abs((new_vals[k] - self.prev_vals[k]) / self.prev_vals[k])).item()
            print(f"{k}: Mean Value: {mean_new_val:.5f}, Absolute Change: {abs_change:.5f}, Relative Change: {rel_change: .5f}")
            all_changes[f"{k}_mean_val"] = mean_new_val
            all_changes[f"{k}_abs"] = abs_change
            all_changes[f"{k}_rel"] = rel_change
            max_abs_change = max(max_abs_change, abs_change)
            max_rel_change = max(max_rel_change, rel_change)

        for k in self.prev_vals:
            self.prev_vals[k] = new_vals[k]

        total_rel_change = min(max_abs_change, max_rel_change)
        all_changes["total"] = total_rel_change
        return all_changes

# ===========================================================================
# Model assembly
# ===========================================================================
def get_model(model_path, K, expert_idx, household_idx, gamma_vec,
              model_size, n_epochs=20000, batch_size=500, lr=1e-3,
              timestepping=False, rar=False, loss_balancing=False,
              params=BASE_PARAMS, train=True, num_outer=70, num_inner=5000,
              min_inner=1000, loss_log_interval=50, max_t=1.0, init_guess=None,
              share_alpha_lo=SHARE_ALPHA_LO, share_alpha_hi=SHARE_ALPHA_HI,
              lr_decay_every=20, lr_decay_gamma=0.5, loss_balancing_alpha=0.9, loss_balancing_temp=0.1, bernoulli_prob=0.99,
              t0_frac=0.4):
    """Assemble (and train if no checkpoint) the heterogeneous N-agent SV model.

    expert_idx / household_idx : 0-based agent indices (their union is 0..K-1).
    gamma_vec                  : length-K risk-aversion vector.
    init_guess                 : optional {name: value} seed for the time-boundary in
                                 time-stepping.
    """
    set_seeds(0)

    # TODO: set the initial batch size to be half for RAR as an experiment
    if rar:
        batch_size = batch_size // 2
    
    if timestepping:
        cfg = {"batch_size": batch_size, "time_batch_size": 1,
               "min_t": 0.0, "max_t": max_t,
               # RARG enables the library's inner-loop residual-refinement hook,
               # which calls self.sample_rar_greedy() and vstacks anchor_points.
               "sampling_method": SamplingMethod.RARG if rar else SamplingMethod.UniformRandom,
               "num_outer_iterations": num_outer, "num_inner_iterations": num_inner,
               "min_inner_iterations": min_inner, "loss_log_interval": loss_log_interval,
               "optimizer_type": OptimizerType.Adam, "lr": lr,
               "loss_balancing": loss_balancing, "rar": rar, "refinement_rounds": 10,
               "share_alpha_lo": share_alpha_lo, "share_alpha_hi": share_alpha_hi,
               "lr_decay_every": lr_decay_every, "lr_decay_gamma": lr_decay_gamma,
               "loss_balancing_alpha": loss_balancing_alpha, "loss_balancing_temp": loss_balancing_temp, "bernoulli_prob": bernoulli_prob,
               # t0-mix training sampler: fraction of each batch pinned to t=min_t
               "t0_frac": t0_frac,
        }
        model = PDEModelTimeStepNAgentsSV("sv_n_agents", cfg)
    else:
        cfg = {"batch_size": batch_size, "num_epochs": n_epochs,
               "sampling_method": SamplingMethod.UniformRandom,
               "optimizer_type": OptimizerType.Adam, "lr": lr,
               "loss_balancing": loss_balancing, "rar": rar, "refinement_rounds": 10,
               "share_alpha_lo": share_alpha_lo, "share_alpha_hi": share_alpha_hi,
               "loss_balancing_alpha": loss_balancing_alpha, "loss_balancing_temp": loss_balancing_temp, "bernoulli_prob": bernoulli_prob,
            }
        model = PDEModelNAgentsSV("sv_n_agents", cfg)

    state_names = [f"x_{i+1}" for i in range(K - 1)] + ["v"]
    domain = {f"x_{i+1}": [0.1 / K, 1 - 0.1 / K] for i in range(K - 1)}
    domain["v"] = list(V_DOMAIN)
    model.set_state(state_names, domain)
    model.add_params(params)
    model.statics = build_statics(K, expert_idx, household_idx, gamma_vec,
                                  has_t=timestepping, params=params)

    # Stacked evaluation requires batch_jac_hes=True, so EVERY network uses it:
    # the stacked evaluator then produces name / name_Jac / name_Hess in a single
    # vmap call, grouping all same-architecture networks together (all the
    # positive xi_k / p share one group; r is separate as it is not positive).
    # xi and p feed the equilibrium through their value AND first/second
    # derivatives; r feeds it by value only (its Jac/Hess are computed but
    # unused), which keeps the config uniform.
    jac_cfg = {"hidden_units": model_size, "batch_jac_hes": True}
    for k in range(1, K + 1):
        model.add_agent(f"xi_{k}", config={**jac_cfg, "positive": True})
    # p is a FREE network (capital price), pinned by the goods-market clearing
    # soft-loss + asset-pricing residual -- exactly as in the original 2-agent
    # model.  (Deriving p analytically from xi instead slaves dp/dt to dxi/dt,
    # which destabilises the backward time-stepping march to the low-p branch.)
    model.add_endog("p", config={**jac_cfg, "positive": True})
    # r is a FREE network (can be negative -> NOT positive).  The asset-pricing
    # residual (added below) is the ONLY equation containing mu_P, so without a
    # free r + this residual the curvature of p is unconstrained.
    model.add_endog("r", config={**jac_cfg})

    xi_names = [f"xi_{k}" for k in range(1, K + 1)]
    model.attach_stacks(xi_names)

    # placeholder entries so equation registration / validation has shapes
    bsz = batch_size
    nm_shp_list = [("goods_resid", (bsz, 1)), ("asset_pricing_resid", (bsz, 1)),
                    ("sig_clearing_resid", (bsz, 1)),
                    ("hjb_expert", (bsz, 1)), ("hjb_household", (bsz, 1)),
                    # extra quantities exposed only for variables_to_track (see
                    # the __check_outer_loop_converge override on the time-step
                    # model); chat/mu_P/r_implied are intermediates.
                    ("chat", (bsz, K)), ("r", (bsz, 1)), ("r_implied", (bsz, 1)),
                    ("mu_P", (bsz, 1)), ("p", (bsz, 1)),
        ]
    for idx in expert_idx:
        nm_shp_list.append((f"hjb_expert_{idx}", (bsz, 1)))
    for idx in household_idx:
        nm_shp_list.append((f"hjb_household_{idx}", (bsz, 1)))
    for nm, shp in nm_shp_list:
        model.variable_val_dict[nm] = torch.zeros(shp, device=model.device)

    # ---- equilibrium-residual losses (spec section 6) ---------------------
    model.add_endog_equation("goods_resid = 0", label="goods")
    model.add_endog_equation("asset_pricing_resid = 0", label="asset_pricing")
    model.add_endog_equation("sig_clearing_resid = 0", label="sig_clearning")
    # hjb_expert/_household are ALREADY sum_k hjb_k**2, so MAE reduction gives
    # mean(sum_k hjb_k**2) == MSE of the raw residual (matching the original).
    # (MSE here would square again -> mean(hjb_k**4), which is wrong.)
    # model.add_hjb_equation("hjb_expert", label="expert", loss_reduction=LossReductionMethod.MAE)
    # model.add_hjb_equation("hjb_household", label="household", loss_reduction=LossReductionMethod.MAE)
    for idx in expert_idx:
        model.add_hjb_equation(f"hjb_expert_{idx}", label=f"expert_{idx}", loss_reduction=LossReductionMethod.MAE)
    for idx in household_idx:
        model.add_hjb_equation(f"hjb_household_{idx}", label=f"household_{idx}", loss_reduction=LossReductionMethod.MAE)

    if train and not os.path.exists(f"{model_path}/model.pt"):
        os.makedirs(model_path, exist_ok=True)
        if timestepping and init_guess:
            # seed the backward march in the correct basin (default guess of 1
            # leaves p pinned ~1 and goods_resid stuck ~1; see get_model docstring)
            model.set_initial_guess(init_guess)
        model.train_model(model_path, "model.pt", full_log=True,
                          variables_to_track=["chat", "r", "r_implied", "p", "mu_P"])
    if os.path.exists(f"{model_path}/model_best.pt"):
        model.load_model(torch.load(f"{model_path}/model_best.pt", weights_only=False,
                                    map_location=device))
        model.attach_stacks(xi_names)
    return model
from __future__ import annotations

import os
import gc

import numpy as np
import pandas as pd
import torch
from torch.func import functional_call, hessian, jacrev, vmap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deep_macrofin import PDEModel, PDEModelTimeStep
from deep_macrofin import OptimizerType, LossReductionMethod, SamplingMethod, set_seeds


# ---------------------------------------------------------------------------
# Fused multi-network derivative helper.
#
# Replaces the library's "one (forward, vmap(jacrev), vmap(hessian)) call per
# agent" pattern with **one fused vmap call across all N agents AND the batch
# dimension**.  At N=50 the per-network GPU-dispatch overhead dominates the
# closure (forward + Jac + Hess for each of N+1 scalar networks => ~3(N+1)
# separate AD graphs).  Stacking the parameters of identical-architecture
# networks and ``functional_call``-ing the shared template lets the GPU run
# **one** kernel per op for all networks at once.
#
# ---------------------------------------------------------------------------
def _stack_module_state_with_grad(modules):
    """Like ``torch.func.stack_module_state`` but **keeps** the autograd graph.

    Returns ``(params, buffers)`` dicts whose stacked tensors share their
    autograd graph with the original ``nn.Parameter``s.  Backprop through the
    stacked tensor distributes per-network gradients into each module's
    parameters in-place (so the existing optimizer keeps working).
    """
    if len(modules) == 0:
        raise ValueError("Need at least one module.")
    all_params = [dict(m.named_parameters()) for m in modules]
    all_buffers = [dict(m.named_buffers()) for m in modules]
    # ``torch.stack`` of leaf Parameters with requires_grad=True produces a
    # non-leaf tensor with ``grad_fn=StackBackward`` -> gradients flow back to
    # each input Parameter's ``.grad`` field.  This is exactly what we need.
    params = {k: torch.stack([p[k] for p in all_params]) for k in all_params[0]}
    buffers = {k: torch.stack([b[k] for b in all_buffers]) for k in all_buffers[0]} if all_buffers[0] else {}
    return params, buffers


class StackedAgent:
    """One vmap-batched call across a list of identical-architecture scalar networks.

    All agents must share the same MLP architecture (matching parameter
    shapes) and produce scalar output (``output_size=1``).

    Public API:
      * ``compute(SV) -> (value, jac, hess)``
          - ``value``: (B, N)
          - ``jac``:   (B, N, D)        where D = number of state variables
          - ``hess``:  (B, N, D, D)
        ``value_only=True`` skips Jac/Hess and returns ``(value, None, None)``.
      * ``value(SV) -> (B, N)``: convenience wrapper.

    No caching: every call re-stacks the current parameters (so any optimizer
    update is picked up) and runs the fused AD.  The stacking is O(N) tiny
    tensor copies, dwarfed by the AD work it enables.
    """

    def __init__(self, agents, value_only: bool = False):
        if len(agents) == 0:
            raise ValueError("StackedAgent needs at least one agent.")
        self._agents = agents
        self._template = agents[0].model
        self._value_only = value_only

    def compute(self, SV: torch.Tensor):
        params, buffers = _stack_module_state_with_grad([a.model for a in self._agents])
        template = self._template

        def value_scalar(p, b, x):
            # (D,) -> scalar (squeeze the single output dim)
            return functional_call(template, (p, b), (x,)).squeeze(-1)

        def fwd_per_net(p, b):
            return vmap(lambda x: value_scalar(p, b, x))(SV)          # (B,)

        val = vmap(fwd_per_net)(params, buffers)                      # (N, B)
        val = val.transpose(0, 1).contiguous()                        # (B, N)

        if self._value_only:
            return val, None, None

        def jac_per_net(p, b):
            return vmap(jacrev(lambda x: value_scalar(p, b, x)))(SV)  # (B, D)

        def hess_per_net(p, b):
            return vmap(hessian(lambda x: value_scalar(p, b, x)))(SV) # (B, D, D)

        jac = vmap(jac_per_net)(params, buffers)                      # (N, B, D)
        hess = vmap(hess_per_net)(params, buffers)                    # (N, B, D, D)
        jac = jac.transpose(0, 1).contiguous()                        # (B, N, D)
        hess = hess.transpose(0, 1).contiguous()                      # (B, N, D, D)
        return val, jac, hess

    def value(self, SV: torch.Tensor):
        return self.compute(SV)[0]


# ---------------------------------------------------------------------------
# Custom sampler (Dirichlet on the (N+1)-simplex with epsilon truncation).
# ---------------------------------------------------------------------------
class PDEModelNAgents(PDEModel):
    def __init__(self, name, config, latex_var_mapping={}):
        super().__init__(name, config, latex_var_mapping)
        self.rar = config.get("rar", False)
        self.n_share = None
        self.n_agents = None
        if self.rar:
            self.sample = self.sample_rar_greedy
            self.sampling_method = SamplingMethod.RARG
        else:
            self.sample = self.sample_simplex
        # disable the per-epoch diagnostic
        self._PDEModel__compute_changes = lambda SV: {"total": 0.0}

        # ---- stacked-agent state ------------------------------------------
        # populated by ``attach_xi_stack`` / ``attach_alpha_stack`` AFTER all ``add_agent`` / ``add_endog`` calls. 
        # Each entry is a triple ``(StackedAgent, active_names)``.
        self._xi_stack: tuple | None = None
        self._alpha_stack: tuple | None = None
        # per-agent forward names whose ``local_function_dict`` entry is now redundant
        self._skip_local_keys: set[str] = set()

    # -- stack attachment ----------------------------------------------------
    def attach_xi_stack(self, active_names):
        """Attach a fused (xi_1, ..., xi_N) stack.

        After this call ``update_variables`` populates
        ``xi_active`` (B, N), ``xi_active_Jac`` (B, N, N), ``xi_active_Hess`` (B, N, N, N)
        in **one** fused vmap call instead of 3*N separate AD calls.
        """
        agents = [self.agents[n] for n in active_names]
        self._xi_stack = (StackedAgent(agents, value_only=False), list(active_names))
        N, D = len(active_names), len(self.state_variables)
        # placeholders -- only used for shape introspection / first-call safety;
        # they are overwritten on every ``update_variables``.
        self.variable_val_dict["xi_active"]      = torch.zeros((self.batch_size, N), device=self.device)
        self.variable_val_dict["xi_active_Jac"]  = torch.zeros((self.batch_size, N, D), device=self.device)
        self.variable_val_dict["xi_active_Hess"] = torch.zeros((self.batch_size, N, D, D), device=self.device)
        self._skip_local_keys.update(active_names)

    def attach_alpha_stack(self, active_names):
        """Attach a fused (alpha_1, ..., alpha_N) stack (value-only)."""
        endogs = [self.endog_vars[n] for n in active_names]
        self._alpha_stack = (StackedAgent(endogs, value_only=True), list(active_names))
        N = len(active_names)
        self.variable_val_dict["alpha_active"] = torch.zeros((self.batch_size, N), device=self.device)
        self._skip_local_keys.update(active_names)

    # -- forward / equation evaluation ---------------------------------------
    def update_variables(self, SV, vd=None):
        """Run one full forward sweep: stacked xi + alpha, then any other
        ``local_function_dict`` entries, then user-defined equations.

        ``vd`` lets callers (notably ``__get_refinement_loss_dict``) write into
        a local copy of the value dict without mutating training state.
        """
        if vd is None:
            vd = self.variable_val_dict

        # 1) fused xi: ONE vmap call computes value+Jac+Hess for all N+1 nets.
        if self._xi_stack is not None:
            stack, active = self._xi_stack
            val, jac, hess = stack.compute(SV)
            N = len(active)
            vd["xi_active"]      = val[:, :N]
            vd["xi_active_Jac"]  = jac[:, :N]
            vd["xi_active_Hess"] = hess[:, :N]
            # keep per-agent names populated for any equation that still
            # references xi_i directly (cheap O(1) tensor slices).
            for i, n in enumerate(active):
                vd[n] = val[:, i:i + 1]

        # 2) fused alpha: forward only (no Jac/Hess needed for portfolio).
        if self._alpha_stack is not None:
            stack, active = self._alpha_stack
            val = stack.value(SV)
            N = len(active)
            vd["alpha_active"] = val[:, :N]
            for i, n in enumerate(active):
                vd[n] = val[:, i:i + 1]

        # 3) remaining ``local_function_dict`` entries (skip the per-agent
        #    forwards already populated above).
        for func_name in self.local_function_dict:
            if func_name in self._skip_local_keys:
                continue
            vd[func_name] = self.local_function_dict[func_name](SV)

        # 4) user-defined equations.
        for eq_name in self.equations:
            lhs = self.equations[eq_name].lhs.formula_str
            vd[lhs] = self.equations[eq_name].eval(self.custom_function_dict, vd)

    def sample_simplex(self, epoch):
        """Dirichlet sample of the N-1 free wealth shares (open simplex)."""
        n_share = self.n_share if getattr(self, "n_share", None) is not None else len(self.state_variables)
        n_agents = self.n_agents if getattr(self, "n_agents", None) is not None else n_share + 1
        eps = 0.02
        max_sum = 0.98
        alpha = torch.ones(n_agents + 1, device=self.device)
        samples = torch.distributions.Dirichlet(alpha).sample((self.batch_size,))
        samples = eps + (max_sum - (n_agents + 1) * eps) * samples
        return samples[:, :n_share]

    def sample_uniform(self, epoch):
        """Override the base cube sampler.
        """
        return self.sample_simplex(epoch)

    def closure(self, SV):
        for i, sv_name in enumerate(self.state_variables):
            self.variable_val_dict[sv_name] = SV[:, i:i + 1]
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
    
    def __get_refinement_loss_dict(self, epoch):
        '''
        Sample a dense subset of the problem domain, compute the loss and return total loss for each point sampled. Used for Residual-based Adaptive Refinement and Active Learning

        Returns:
            {
                "SV": sampled state variables, shape (10000, len(self.state_variables))
                "loss": total loss computed at each sv, shape (10000, 1)
            }
        '''
        # because we need a set of dense points to compute residual for adaptive sampling
        # we set all models to evaluation models so that gradients won't be computed.
        # it speeds up the computation and reduces memory usages
        self.set_all_model_eval()

        # Temporarily set a large batch size
        all_SVs = []
        all_total_loss = []
        self.batch_size = 1000
        for _ in range(10):
            torch.cuda.empty_cache()
            SV = self.sample_simplex(epoch)
            SV.requires_grad_(True)
            # make a copy of variable value mapping
            # so that we don't break the top level training routine
            variable_val_dict_ = self.variable_val_dict.copy()
            total_loss = torch.zeros((self.batch_size, 1), device=self.device)

            # forward pass
            for i, sv_name in enumerate(self.state_variables):
                variable_val_dict_[sv_name] = SV[:, i:i+1]
            variable_val_dict_["SV"] = SV

            # update variables (stacked xi + alpha, remaining local fns, equations)
            # routed through the shared override so RAR sees the same fused
            # forward as training does.
            self.update_variables(SV, vd=variable_val_dict_)

            # Sum per-sample |residual| across every loss component into a
            # (B, 1) score.  Residuals come back in several shapes:
            #   * (B, 1)      most equations
            #   * (B, k)      multi-output residuals like vi_active
            #                 (one entry per active agent)
            #   * (B,)        1-D per-sample tensors
            #   * 0-D scalar  pre-reduced HJB (fine to broadcast)
            # We collapse the trailing dims by mean so the per-point score
            # remains a balanced sum across loss components.
            B = SV.shape[0]

            def _as_per_sample(resid_):
                a = torch.abs(resid_)
                if a.dim() == 0:
                    return a.expand(B, 1).reshape(B, 1)
                if a.dim() == 1:
                    return a.reshape(B, 1)
                return a.reshape(B, -1).mean(dim=-1, keepdim=True)

            for label in self.endog_equations:
                resid = self.endog_equations[label].eval_no_loss(self.custom_function_dict, variable_val_dict_)
                total_loss = total_loss + _as_per_sample(resid)

            for label in self.constraints:
                resid = self.constraints[label].eval_no_loss(self.custom_function_dict, variable_val_dict_)
                total_loss = total_loss + _as_per_sample(resid)

            for label in self.hjb_equations:
                resid = self.hjb_equations[label].eval_no_loss(self.custom_function_dict, variable_val_dict_)
                total_loss = total_loss + _as_per_sample(resid)

            for label in self.systems:
                resid = self.systems[label].eval_no_loss(self.custom_function_dict, variable_val_dict_, self.batch_size)
                total_loss = total_loss + _as_per_sample(resid)

            all_SVs.append(SV.detach().cpu())
            all_total_loss.append(total_loss.detach().cpu())
            del SV, total_loss
            gc.collect()
            torch.cuda.empty_cache()

        self.batch_size = self.config.get("batch_size", 100) # reset the batch size for normal computation
        self.set_all_model_training() # reset the model for training stage

        return {
            "SV": torch.cat(all_SVs, dim=0),
            "loss": torch.cat(all_total_loss, dim=0),
        }
        
    def sample_rar_greedy(self, epoch):
        inner = getattr(self, "num_inner_iterations", None) or self.num_epochs
        if epoch % max(1, inner // self.refinement_rounds) == 0 and epoch > 0:
            refinement_loss_dict = self.__get_refinement_loss_dict(epoch)
            SV = refinement_loss_dict["SV"]
            all_losses = refinement_loss_dict["loss"]
            X_ids = torch.topk(all_losses, self.batch_size//self.refinement_rounds, dim=0)[1].squeeze(-1)
            picked = SV[X_ids].to(self.device)
            if self.anchor_points is None:
                self.anchor_points = picked
            else:
                self.anchor_points = torch.vstack((self.anchor_points, picked))
        sv = self.sample_simplex(epoch)
        if self.anchor_points is not None and len(self.anchor_points) > 0:
            return torch.vstack((sv, self.anchor_points))
        return sv
    
    def load_model(self, dict_to_load):
        # CRITICAL for the stacked model + time-stepping outer loop:
        # the library's load_model (called at the end of every outer loop)
        # rebuilds each agent/endog var via add_agent(overwrite=True), which
        # constructs BRAND-NEW module objects and rebinds self.agents[name] /
        # self.endog_vars[name].  Our StackedAgents hold direct references to the
        # *previous* objects, so without re-attaching, the reinitialized
        # optimizer would train the new modules while update_variables still
        # evaluates the old (frozen) ones -- the model then stops changing after
        # the first time iteration.
        super().load_model(dict_to_load)
        xi_active_names    = [f"xi_{i+1}"    for i in range(self.n_agents)]
        alpha_active_names = [f"alpha_{i+1}" for i in range(self.n_agents)]
        self.attach_xi_stack(xi_active_names)
        self.attach_alpha_stack(alpha_active_names)


class PDEModelTimeStepNAgents(PDEModelNAgents, PDEModelTimeStep):
    """Time-stepping GP model: pseudo-time ``t`` is appended to the state.

    Only drift terms pick up ``mu_t = 1``; volatilities and wealth-share
    economics are unchanged (see ``compute_mu_state_n``).
    """

    def __init__(self, name, config, latex_var_mapping={}):
        PDEModelTimeStep.__init__(self, name, config, latex_var_mapping)
        self.rar = config.get("rar", False)
        self.n_share = None
        self.n_agents = None
        self._xi_stack = None
        self._alpha_stack = None
        self._skip_local_keys = set()
        try:
            self._PDEModel__compute_changes = lambda SV: {"total": 0.0}
        except Exception:
            pass
        if self.rar:
            self.sample = self.sample_rar_greedy_ts
            self.sampling_method = SamplingMethod.RARG
        else:
            self.sample = self.sample_simplex_ts
        self.sample_boundary_cond = self.__sample_custom_boundary_cond
        self.boundary_uniform_points = None

    def sample_uniform(self, epoch):
        return self.sample_simplex_ts(epoch)

    def sample_simplex_ts(self, epoch=0):
        base = self.sample_simplex(epoch)
        min_t = self.config.get("min_t", 0.0)
        max_t = self.config.get("max_t", 1.0)
        t = min_t + (max_t - min_t) * torch.rand((base.shape[0], 1), device=self.device)
        return torch.cat([base, t], dim=1)
    
    def __sample_custom_boundary_cond(self, time_val: float):
        if self.boundary_uniform_points is None:
            self.boundary_uniform_points = self.sample_simplex(0)
        time_dim = torch.ones((self.boundary_uniform_points.shape[0], 1), device=self.device) * time_val
        return torch.cat([self.boundary_uniform_points, time_dim], dim=-1)

    def sample_rar_greedy_ts(self, epoch=0):
        inner = self.num_inner_iterations
        if epoch % max(1, inner // self.refinement_rounds) == 0 and epoch > 0:
            self.set_all_model_eval()
            SVs, losses = [], []
            saved_bs = self.batch_size
            self.batch_size = 1000
            for _ in range(5):
                SV = self.sample_simplex_ts(epoch)
                SV.requires_grad_(True)
                vd_ = self.variable_val_dict.copy()
                for i, sv_name in enumerate(self.state_variables):
                    vd_[sv_name] = SV[:, i:i + 1]
                vd_["SV"] = SV
                self.update_variables(SV, vd=vd_)
                total = torch.zeros((SV.shape[0], 1), device=self.device)
                Bn = SV.shape[0]
                for label in self.hjb_equations:
                    res = torch.abs(self.hjb_equations[label].eval_no_loss(self.custom_function_dict, vd_))
                    total = total + (res.reshape(Bn, -1).mean(dim=-1, keepdim=True) if res.dim() > 1 else res.reshape(Bn, 1))
                SVs.append(SV.detach().cpu())
                losses.append(total.detach().cpu())
                del SV, total
            self.batch_size = saved_bs
            self.set_all_model_training()
            SVall = torch.cat(SVs, 0)
            lall = torch.cat(losses, 0)
            ids = torch.topk(lall, self.batch_size // self.refinement_rounds, dim=0)[1].squeeze(-1)
            if self.anchor_points is None:
                self.anchor_points = SVall[ids].to(self.device)
            else:
                self.anchor_points = torch.vstack((self.anchor_points, SVall[ids].to(self.device)))
        sv = self.sample_simplex_ts(epoch)
        if self.anchor_points is not None and len(self.anchor_points) > 0:
            return torch.vstack((sv, self.anchor_points))
        return sv


device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Closed-form y, dy/dx, d2y/dx2 on the (N-1)-simplex  {sum_i x_i = 1}.
#
# There are N active agents whose wealth shares sum to 1.  We parametrise the
# state by the N-1 FREE shares ``SV = (x_1, ..., x_{N-1})`` and treat agent N
# as the DEPENDENT coordinate ``x_N = 1 - sum_{j<N} x_j`` (exactly the role the
# passive agent x_p played in the 3-agent base).  Because x_N depends on every
# free coordinate (dx_N/dx_j = -1), the closed-form derivatives carry the extra
# ``- xi_N`` / ``- J_{N,*}`` terms below.
#
# Inputs:
#   SV              (B, N-1)         free shares x_1, ..., x_{N-1}
#   xi_active       (B, N)           xi_i  (all N agents)
#   xi_active_Jac   (B, N, N-1)      d xi_i / d x_j   (j over free coords)
#   xi_active_Hess  (B, N, N-1, N-1)
# Outputs:
#   y               (B, 1)
#   dy_dx           (B, 1, N-1)
#   d2y_dx2         (B, 1, N-1, N-1)
# ---------------------------------------------------------------------------
def compute_x_full(SV):
    """Append the dependent share x_N = 1 - sum_{j<N} x_j -> (B, N)."""
    x_dep = 1.0 - SV.sum(dim=-1, keepdim=True)
    return torch.cat([SV, x_dep], dim=-1)


def compute_y_closed(SV, xi_active):
    x = compute_x_full(SV)                                 # (B, N)
    return (x * xi_active).sum(dim=-1, keepdim=True)


def compute_dy_dx_closed(SV, xi_active, xi_active_Jac):
    x = compute_x_full(SV)                                 # (B, N)
    xi_free = xi_active[:, :-1]                            # (B, N-1)
    xi_dep = xi_active[:, -1:]                             # (B, 1)
    # dy/dx_j = (xi_j - xi_N) + sum_i x_i J_{i,j}
    dy = (xi_free - xi_dep) + torch.einsum("bi,bij->bj", x, xi_active_Jac)
    return dy.unsqueeze(1)                                 # (B, 1, N-1)


def compute_d2y_dx2_closed(SV, xi_active, xi_active_Jac, xi_active_Hess):
    x = compute_x_full(SV)                                 # (B, N)
    J_free = xi_active_Jac[:, :-1, :]                      # (B, N-1, N-1)  agents < N
    J_dep = xi_active_Jac[:, -1, :]                        # (B, N-1)       agent N
    sym_jac = J_free + J_free.transpose(-1, -2)            # J_{j,k} + J_{k,j}
    dep_term = J_dep.unsqueeze(1) + J_dep.unsqueeze(2)     # J_{N,k} + J_{N,j}
    hess_term = torch.einsum("bi,bijk->bjk", x, xi_active_Hess)
    d2y = sym_jac - dep_term + hess_term                   # (B, N-1, N-1)
    return d2y.unsqueeze(1)                                # (B, 1, N-1, N-1)


# ---------------------------------------------------------------------------
# N-agent generalisations of the GP model primitives.
# ---------------------------------------------------------------------------
def compute_sigy_n(SV, y, dy_dx, alpha_active, sigma):
    """
    A = sum_{j<N} (dy/dx_j) * x_j * (alpha_j - 1)   (sum over FREE states)
    sigy = A * sigma / (y + A)
    (single-shock economy; dependent agent N carries no own state)
    """
    y_x = dy_dx[:, 0, :]                                   # (B, N-1)
    alpha_free = alpha_active[:, :-1]                      # (B, N-1)
    A = (y_x * SV * (alpha_free - 1.0)).sum(dim=-1, keepdim=True)
    return (A * sigma.reshape(1, 2)) / (y + A + 1e-8)


def compute_sigx_n(SV, alpha_active, sigR):
    """sigx[b, j, :] = x_j * (alpha_j - 1) * sigR for the FREE states j<N;
    shape (B, N-1, 2)."""
    alpha_free = alpha_active[:, :-1]                      # (B, N-1)
    coef = SV * (alpha_free - 1.0)                         # (B, N-1)
    return coef.unsqueeze(-1) * sigR.unsqueeze(1)          # (B, N-1, 2)


def compute_varsigma_active(gamma_active, psi_active, sigxi_active, sigR):
    """varsigma_i = (1 - 1/gamma_i)/(1 - psi_i) * (sigxi_i . sigR) / |sigR|^2.

    gamma_active, psi_active are 1-D tensors of shape (N,).
    sigxi_active has shape (B, N, 2);  sigR has shape (B, 2).
    """
    sigR_sq = (sigR ** 2).sum(dim=-1, keepdim=True)        # (B, 1)
    dot = (sigxi_active * sigR.unsqueeze(1)).sum(dim=-1)   # (B, N)
    coef = ((1.0 - 1.0 / gamma_active) / (1.0 - psi_active + 1e-8)).reshape(1, -1)
    return coef * dot / (sigR_sq + 1e-8)                   # (B, N)


def compute_mux_n(SV, xi_active, alpha_active, y, q, sigR, kappa, omega_active):
    """Drift of the FREE state shares x_1, ..., x_{N-1};  shape (B, N-1)."""
    sigR_sq = (sigR ** 2).sum(dim=-1, keepdim=True)        # (B, 1)
    xi_free = xi_active[:, :-1]                            # (B, N-1)
    alpha_free = alpha_active[:, :-1]                      # (B, N-1)
    omega_free = omega_active.reshape(1, -1)[:, :SV.shape[1]]
    return SV * (y - xi_free + (1.0 - alpha_free) * (1.0 - q) * sigR_sq)  + kappa * (omega_free - SV)


def compute_a_mat_n(sigx_active):
    """a_mat = sigx @ sigx^T  in shock dim;  shape (B, N, N)."""
    return torch.einsum("bij,bkj->bik", sigx_active, sigx_active)


def compute_share_sv(SV, n_share):
    """Wealth-share block: first ``n_share`` columns (drops pseudo-time t)."""
    n = int(n_share)
    return SV[:, :n]


def compute_mu_state_n(SV, mux):
    """Full state drift: wealth-share drifts plus ``mu_t = 1`` when t is present."""
    n_share = mux.shape[-1]
    D = SV.shape[-1]
    if D == n_share:
        return mux
    mu_state = torch.zeros(SV.shape[0], D, device=SV.device, dtype=SV.dtype)
    mu_state[:, :n_share] = mux
    mu_state[:, -1] = 1.0
    return mu_state


def compute_a_mat_state_n(sigx_active, SV, mux):
    """Diffusion matrix on the full state; t has zero diffusion."""
    a_share = compute_a_mat_n(sigx_active)
    n_share = mux.shape[-1]
    D = SV.shape[-1]
    if D == n_share:
        return a_share
    B = SV.shape[0]
    a_full = torch.zeros(B, D, D, device=SV.device, dtype=SV.dtype)
    a_full[:, :n_share, :n_share] = a_share
    return a_full


def compute_muxi_active(xi_active, xi_active_Jac, xi_active_Hess, mux, a_mat):
    """muxi_k = (1/xi_k)[ J[k, .] . mux  +  0.5 * tr(H[k, ., .] . a_mat) ]."""
    drift = torch.einsum("bij,bj->bi", xi_active_Jac, mux) / (xi_active + 1e-8)
    diff = 0.5 * torch.einsum("bijk,bkj->bi", xi_active_Hess, a_mat) / (xi_active + 1e-8)
    return drift + diff                                    # (B, N)


def compute_muy_n(y, dy_dx, d2y_dx2, mux, a_mat, SV=None, xi_active=None, xi_active_Jac=None):
    """Aggregate consumption-wealth drift.

    For time-stepping, ``mux`` / ``a_mat`` are the full-state objects
    (with ``mu_t = 1`` on the last coordinate).  The share-block Ito terms
    use the first ``N-1`` coordinates; the explicit ``t`` contribution enters
    through ``sum_i x_i (d xi_i / dt)`` with ``mu_t = 1``.
    """
    n_share = dy_dx.shape[-1]
    mux_share = mux[:, :n_share]
    a_share = a_mat[:, :n_share, :n_share]
    drift = torch.einsum("bij,bj->bi", dy_dx, mux_share) / (y + 1e-8)
    diff = 0.5 * torch.einsum("bijk,bkj->bi", d2y_dx2, a_share) / (y + 1e-8)
    out = drift + diff
    if mux.shape[-1] > n_share and SV is not None and xi_active_Jac is not None:
        x = compute_x_full(SV[:, :n_share])
        out = out + (x * xi_active_Jac[:, :, n_share]).sum(dim=-1, keepdim=True) / (y + 1e-8)
    return out


def compute_muP_n(muy, sigy, sigma, mu):
    return mu - muy + (sigy * (sigy - sigma.reshape(1, 2))).sum(dim=-1, keepdim=True)


def compute_hjb_n(gamma_active, psi_active, muxi_active, sigxi_active, sigR,
                  r, eta, alpha_active, sigR_norm,
                  xi_active, rho):
    """Aggregated HJB residual as a *per-sample* tensor of shape ``(B, 1)``.

    Each entry is ``sum_k hjb_k(b)**2`` (squared residual summed over all
    N+1 agents at batch index ``b``).  Registering the HJB equation with
    ``LossReductionMethod.MAE`` then averages over the batch, yielding the
    *same* scalar training loss as the previous "MSE over each agent then
    sum over agents" reduction:

        (1/B) * sum_b sum_k hjb_k(b)^2.

    Returning per-sample is what makes residual-adaptive sampling (RAR) work:
    ``eval_no_loss`` now gives the refinement loop a non-trivial per-point
    HJB residual instead of a single broadcast scalar.
    """
    # active agents
    ga = gamma_active.reshape(1, -1)
    pa = psi_active.reshape(1, -1)
    hjb_active = (
        rho * pa
        + (1 - pa) * (r + eta * alpha_active * sigR_norm
                      - ga / 2 * (alpha_active * sigR_norm) ** 2)
        + muxi_active
        + (1 - ga) * (sigxi_active * sigR.unsqueeze(1)).sum(dim=-1) * alpha_active
        + 0.5 * (pa - ga) / (1 - pa + 1e-8) * (sigxi_active ** 2).sum(dim=-1)
        - xi_active
    ) / rho
    res_a = (hjb_active ** 2).sum(dim=-1, keepdim=True)        # (B, 1)
    return res_a


def compute_market_clearing_n(SV, alpha_active):
    x = compute_x_full(SV)                                 # (B, N)
    return (x * alpha_active).sum(dim=-1, keepdim=True)


def compute_foc_active(q, gamma_active, varsigma_active):
    return q / gamma_active.reshape(1, -1) - varsigma_active


def compute_pricing_n(SV, varsigma_active, sigR_norm, gamma_active):
    """pi = (1 + sum_i x_i varsigma_i) |sigR|^2 / (sum_i x_i/gamma_i),
    summed over ALL N agents (x_N is the dependent share)."""
    x = compute_x_full(SV)                                 # (B, N)
    NUM = (1.0 + (x * varsigma_active).sum(dim=-1, keepdim=True)) * sigR_norm ** 2
    DEN = (x / gamma_active.reshape(1, -1)).sum(dim=-1, keepdim=True)
    return NUM / (DEN + 1e-8)


# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------
def get_model(model_path: str, n_active: int, params: dict,
              model_size: list[int], n_epochs: int = 8000,
              batch_size: int = 200, lr: float = 5e-4,
              alpha_caps=None,
              timestepping: bool = False,
              rar: bool = False,
              loss_balancing: bool = False,
              train: bool = True,
              num_outer: int = 70, num_inner: int = 5000,
              min_inner: int = 1000, max_t: float = 1,
              init_guess: dict | None = None):
    """Assemble and (if no checkpoint) train the heterogeneous N-agent model.

    There are ``n_active = N`` active agents whose wealth shares sum to 1.  The
    state is the (N-1)-simplex: the free shares ``x_1, ..., x_{N-1}`` with the
    dependent share ``x_N = 1 - sum_{j<N} x_j``.  Every agent (including agent
    N) has its own ``xi_i`` and ``alpha_i`` network, HJB and portfolio FOC.

    ``params`` must contain (all torch.Tensors on ``device`` unless noted):
        - gamma_active (N,) 
        - psi_active (N,)
        - omega_active (N,)
        - rho, mu, sigma (2,), kappa

    Free-boundary constraints
    -------------------------
    ``alpha_caps`` is a length-N iterable.  Entry i = +inf (or any value
    >= 1e3) marks agent i as **unconstrained**; a finite entry imposes the
    cap ``alpha_i <= alpha_caps[i]`` via the variational inequality

        min(alpha_caps[i] - alpha_i,  q/gamma_i - varsigma_i - alpha_i) = 0.

    The min reduces to the ordinary Merton FOC residual when the cap is
    inactive (cap - alpha is very large), and to the binding constraint
    (alpha = cap) when the FOC pushes alpha above the ceiling.  This is
    *exactly* the residual used in ``gp_constraint_NN.py`` for the c-agent,
    generalised to per-agent caps.

    .. note::
       Agent 1 is the anchor for ``q = gamma_1 * (alpha_1 + varsigma_1)``.
       Its FOC residual is identically zero by construction, so it *cannot*
       be a constrained agent.  Put any capped agents at indices 2..N.
    """
    set_seeds(42)
    if timestepping:
        cfg = {
            "batch_size": batch_size, "time_batch_size": 1,
            "min_t": 0.0, "max_t": max_t,
            "sampling_method": SamplingMethod.UniformRandom,
            "num_outer_iterations": num_outer, "num_inner_iterations": num_inner,
            "min_inner_iterations": min_inner, "loss_log_interval": 50,
            "optimizer_type": OptimizerType.Adam, "lr": lr,
            "rar": rar, "refinement_rounds": 10,
            "loss_balancing": loss_balancing,
        }
        model = PDEModelTimeStepNAgents("gp_n_agents", config=cfg)
    else:
        cfg = {
            "batch_size": batch_size, "num_epochs": n_epochs,
            "optimizer_type": OptimizerType.Adam, "lr": lr,
            "rar": rar, "refinement_rounds": 10,
            "loss_balancing": loss_balancing,
        }
        model = PDEModelNAgents("gp_n_agents", config=cfg)

    # state = N-1 FREE wealth shares [+ pseudo-time t]; x_N = 1 - sum is dependent
    n_state = n_active - 1
    model.n_share = n_state
    model.n_agents = n_active
    state_names = [f"x_{i+1}" for i in range(n_state)]
    domain = {nm: [0.0, 1.0] for nm in state_names}
    model.set_state(state_names, domain)

    # ---- register alpha_caps as a model parameter (broadcast-ready) -----
    if alpha_caps is None:
        alpha_caps = [1e6] * n_active                       # all unconstrained
    alpha_caps_arr = np.asarray(alpha_caps, dtype=np.float32).reshape(-1)
    assert alpha_caps_arr.shape == (n_active,), f"alpha_caps must have shape ({n_active},), got {alpha_caps_arr.shape}"
    assert alpha_caps_arr[0] >= 1e3,  ("Agent 1 anchors q := gamma_1 * (alpha_1 + varsigma_1); it must be "
         "unconstrained (alpha_caps[0] >= 1e3).  Put constrained agents at "
         "indices 2..N.")
    # cap tensor shape (1, N) so broadcast against (B, N) is unambiguous
    params = dict(params)
    params["alpha_caps"] = torch.tensor(alpha_caps_arr, device=device, dtype=torch.get_default_dtype()).reshape(1, -1)
    model.add_params(params)

    model.register_functions([
        compute_y_closed, compute_dy_dx_closed, compute_d2y_dx2_closed,
        compute_sigy_n, compute_sigx_n,
        compute_varsigma_active,
        compute_mux_n, compute_a_mat_n,
        compute_share_sv, compute_mu_state_n, compute_a_mat_state_n,
        compute_muxi_active,
        compute_muy_n, compute_muP_n, compute_hjb_n,
        compute_market_clearing_n,
        compute_foc_active, compute_pricing_n,
    ])

    # ---- per-agent active networks (xi_i and alpha_i for i = 1..N) ------
    for i in range(1, n_active + 1):
        model.add_agent(f"xi_{i}", config={"hidden_units": model_size, "derivative_order": 0, "batch_jac_hes": False, "positive": True})
        model.add_endog(f"alpha_{i}", config={"hidden_units": model_size, "derivative_order": 0, "batch_jac_hes": False})
    # ---- attach fused StackedAgent over all active xi and active alpha.
    # After this:
    #   * xi_active   (B, N)
    #   * xi_active_Jac  (B, N, N-1)
    #   * xi_active_Hess (B, N, N-1, N-1)
    #   * alpha_active (B, N)
    # are populated by ``update_variables`` in *one* fused vmap call each,
    # replacing the N+1 per-network forward/Jac/Hess calls the library would
    # otherwise dispatch sequentially.
    xi_active_names    = [f"xi_{i+1}"    for i in range(n_active)]
    alpha_active_names = [f"alpha_{i+1}" for i in range(n_active)]
    model.attach_xi_stack(xi_active_names)
    model.attach_alpha_stack(alpha_active_names)

    if timestepping:
        # The xi networks take (x_1, ..., x_{N-1}, t), so xi_active_Jac is
        # (B, N, N) and xi_active_Hess is (B, N, N, N) -- the LAST column/slab
        # is the pseudo-time derivative.  The closed-form y, dy/dx, d2y/dx2 and
        # all *spatial* objects (sigy, sigx, sigxi, the Ito diffusion of y/xi)
        # must use ONLY the wealth-share block of the Jacobian/Hessian; mixing
        # in the t-column corrupts the spatial derivatives.  The pseudo-time
        # contribution re-enters drift terms (muxi, muy) explicitly through
        # ``mu_state`` (which carries mu_t = 1 on the last coordinate).
        Js = f"xi_active_Jac[:, :, :{n_state}]"
        Hs = f"xi_active_Hess[:, :, :{n_state}, :{n_state}]"
        eqs = [
            f"SV_share=compute_share_sv(SV, {n_state})",
            f"xi_Jac_s={Js}",
            f"xi_Hess_s={Hs}",
            "y=compute_y_closed(SV_share, xi_active)",
            "dy_dx=compute_dy_dx_closed(SV_share, xi_active, xi_Jac_s)",
            "d2y_dx2=compute_d2y_dx2_closed(SV_share, xi_active, xi_Jac_s, xi_Hess_s)",
            "sigy=compute_sigy_n(SV_share, y, dy_dx, alpha_active, sigma)",
            "sigR=sigma.reshape(1,2)-sigy",
            "sigR_norm=torch.sqrt(torch.sum(sigR ** 2, dim=1, keepdim=True))",
            "sigx_active=compute_sigx_n(SV_share, alpha_active, sigR)",
            "sigxi_active=torch.bmm(xi_Jac_s/(xi_active.unsqueeze(-1)+1e-8), sigx_active)",
            "varsigma_active=compute_varsigma_active(gamma_active, psi_active, sigxi_active, sigR)",
            "q=gamma_active[0]*(alpha_active[:, 0:1]+varsigma_active[:, 0:1])",
            "pi=q*sigR_norm**2",
            "eta=q*sigR_norm",
            "mux=compute_mux_n(SV_share, xi_active, alpha_active, y, q, sigR, kappa, omega_active)",
            # full-state drift (mu_t = 1 on the t-coord) and diffusion (zero on t)
            "mu_state=compute_mu_state_n(SV, mux)",
            "a_mat=compute_a_mat_state_n(sigx_active, SV, mux)",
            # All Ito drifts carry the pseudo-time derivative (matches the SV
            # model): muxi picks up d(xi)/dt, muy picks up d(y)/dt.  Only the
            # *spatial* (share-block) Jacobian/Hessian enter the closed-form
            # derivatives above; the t-contribution is injected here via mu_state.
            "muxi_active=compute_muxi_active(xi_active, xi_active_Jac, xi_active_Hess, mu_state, a_mat)",
            "muy=compute_muy_n(y, dy_dx, d2y_dx2, mu_state, a_mat, SV, xi_active, xi_active_Jac)",
            "muP=compute_muP_n(muy, sigy, sigma, mu)",
            "r=y+muP-pi",
            "hjb=compute_hjb_n(gamma_active, psi_active, muxi_active, sigxi_active, sigR, r, eta, alpha_active, sigR_norm, xi_active, rho)",
            "foc_active_target=compute_foc_active(q, gamma_active, varsigma_active)",
            "pricing_target=compute_pricing_n(SV_share, varsigma_active, sigR_norm, gamma_active)",
        ]
    else:
        eqs = [
            "y=compute_y_closed(SV, xi_active)",
            "dy_dx=compute_dy_dx_closed(SV, xi_active, xi_active_Jac)",
            "d2y_dx2=compute_d2y_dx2_closed(SV, xi_active, xi_active_Jac, xi_active_Hess)",
            "sigy=compute_sigy_n(SV, y, dy_dx, alpha_active, sigma)",
            "sigR=sigma.reshape(1,2)-sigy",
            "sigR_norm=torch.sqrt(torch.sum(sigR ** 2, dim=1, keepdim=True))",
            "sigx_active=compute_sigx_n(SV, alpha_active, sigR)",
            f"sigxi_active=torch.bmm((xi_active_Jac/(xi_active.unsqueeze(-1)+1e-8))[:,:,:{n_state}], sigx_active)",
            "varsigma_active=compute_varsigma_active(gamma_active, psi_active, sigxi_active, sigR)",
            "q=gamma_active[0]*(alpha_active[:, 0:1]+varsigma_active[:, 0:1])",
            "pi=q*sigR_norm**2",
            "eta=q*sigR_norm",
            "mux=compute_mux_n(SV, xi_active, alpha_active, y, q, sigR, kappa, omega_active)",
            "a_mat=compute_a_mat_n(sigx_active)",
            "muxi_active=compute_muxi_active(xi_active, xi_active_Jac, xi_active_Hess, mux, a_mat)",
            "muy=compute_muy_n(y, dy_dx, d2y_dx2, mux, a_mat)",
            "muP=compute_muP_n(muy, sigy, sigma, mu)",
            "r=y+muP-pi",
            "hjb=compute_hjb_n(gamma_active, psi_active, muxi_active, sigxi_active, sigR, r, eta, alpha_active, sigR_norm, xi_active, rho)",
            "foc_active_target=compute_foc_active(q, gamma_active, varsigma_active)",
            "pricing_target=compute_pricing_n(SV, varsigma_active, sigR_norm, gamma_active)",
        ]
    model.add_equations(eqs)

    # ---- equilibrium residuals ------------------------------------------
    mc_sv = "SV_share" if timestepping else "SV"
    model.add_endog_equation(f"compute_market_clearing_n({mc_sv}, alpha_active)=1", label="mc")
    # Per-agent variational inequality (NCP form).  Reduces to the plain
    # Merton FOC residual when alpha_caps[i] is large (unconstrained), and
    # enforces alpha_i = cap_i when the constraint binds.  Broadcasts cleanly
    # because alpha_caps has shape (1, N) and alpha_active has shape (B, N).
    model.add_endog_equation("torch.minimum(alpha_caps - alpha_active, foc_active_target - alpha_active)=0", label="vi_active")
    # Walrasian pricing equation (over-identifying check on q).
    model.add_endog_equation("pi=pricing_target", label="pricing")
    # HJB residual. 
    model.add_hjb_equation("hjb", loss_reduction=LossReductionMethod.MAE)

    if timestepping and init_guess is not None:
        model.set_initial_guess({k: v for k, v in init_guess.items()
                                 if k in model.agents or k in model.endog_vars})

    if train and not os.path.exists(f"{model_path}/model.pt"):
        os.makedirs(model_path, exist_ok=True)
        model.train_model(model_path, "model.pt", True)
    if os.path.exists(f"{model_path}/model_best.pt"):
        model.load_model(torch.load(f"{model_path}/model_best.pt", weights_only=False))
        # ``load_model`` re-creates each agent / endog via ``add_agent(..., overwrite=True)``,
        # so the StackedAgent's cached ``agents[i].model`` references are now stale.
        # Re-attach so the post-load forward sweep uses the freshly-loaded modules.
        model.attach_xi_stack(xi_active_names)
        model.attach_alpha_stack(alpha_active_names)
    return model


# ---------------------------------------------------------------------------
# Diagnostics: evaluate the trained NN on a slice of state space
# ---------------------------------------------------------------------------
def _state_for_model(model: PDEModel, SV_share: torch.Tensor) -> torch.Tensor:
    """Append ``t = min_t`` when the model uses pseudo-time stepping."""
    if "t" in model.state_variables and SV_share.shape[-1] == getattr(model, "n_share", len(model.state_variables) - 1):
        min_t = model.config.get("min_t", 0.0)
        t = torch.full((SV_share.shape[0], 1), min_t, device=SV_share.device, dtype=SV_share.dtype)
        return torch.cat([SV_share, t], dim=-1)
    return SV_share


def evaluate_along(model: PDEModel, SV_np: np.ndarray):
    """Evaluate model variables at the given free-share points.

    ``SV_np`` may be ``(B, N-1)`` wealth shares only, or already include ``t``
    as the last column for time-stepping models.
    """
    n_share = getattr(model, "n_share", None)
    if n_share is None:
        n_share = len(model.state_variables) - (1 if "t" in model.state_variables else 0)
    share_np = SV_np[:, :n_share]
    if "t" in model.state_variables and SV_np.shape[-1] == n_share:
        min_t = model.config.get("min_t", 0.0)
        SV_np = np.concatenate([share_np, np.full((share_np.shape[0], 1), min_t, dtype=share_np.dtype)], axis=1)

    SV = torch.tensor(SV_np, device=model.device, dtype=torch.get_default_dtype())
    SV.requires_grad_(True)
    for i, sv_name in enumerate(model.state_variables):
        model.variable_val_dict[sv_name] = SV[:, i:i + 1]
    model.variable_val_dict["SV"] = SV
    model.update_variables(SV)

    n_state = n_share
    alpha_active = model.variable_val_dict["alpha_active"].detach().cpu().numpy()  # (B, N)
    n_agents = alpha_active.shape[1]
    x_dep = 1.0 - share_np.sum(axis=-1)                       # dependent share x_N
    x_full = np.concatenate([share_np, x_dep[:, None]], axis=1)   # (B, N)
    out = {
        "r":   model.variable_val_dict["r"].detach().cpu().numpy().reshape(-1),
        "pi":  model.variable_val_dict["pi"].detach().cpu().numpy().reshape(-1),
        "y":   model.variable_val_dict["y"].detach().cpu().numpy().reshape(-1),
        "q":   model.variable_val_dict["q"].detach().cpu().numpy().reshape(-1),
        "x_total": x_full.sum(axis=-1),                    # identically 1 on the simplex
    }
    for i in range(n_agents):
        out[f"x_{i+1}"] = x_full[:, i]
        out[f"xi_{i+1}"] = model.variable_val_dict[f"xi_{i+1}"].detach().cpu().numpy().reshape(-1)
        out[f"alpha_{i+1}"] = alpha_active[:, i]
    return pd.DataFrame(out)


def _build_x1_path(n_active: int, x_grid: np.ndarray) -> np.ndarray:
    """Free-coordinate array ``(B, N-1)`` for a 1-D slice that varies agent 1's
    wealth share ``x_1 = x_grid`` and splits the remaining ``1 - x_1`` equally
    among the other N-1 agents, so ``x_2 = ... = x_N = (1 - x_1)/(N-1)``.

    On the {sum_i x_i = 1} simplex this is the natural 1-D cut; for N=2 it is
    simply ``x_1 in (0,1)`` with the dependent share ``x_2 = 1 - x_1``.
    """
    x_grid = np.asarray(x_grid, dtype=np.float32).reshape(-1)
    n_state = n_active - 1
    SV = np.empty((x_grid.shape[0], n_state), dtype=np.float32)
    SV[:, 0] = x_grid
    if n_state > 1:
        SV[:, 1:] = ((1.0 - x_grid) / (n_active - 1))[:, None]
    return SV


def plot_symmetric_slice(model: PDEModel, n_active: int, output_dir: str,
                         reference_csv: str = None, alpha_caps=None):
    """Evaluate along the x_1 cut (x_2 = ... = x_N = (1-x_1)/(N-1)) and overlay
    the Chebyshev reference, plotting against agent 1's wealth share x_1."""
    os.makedirs(output_dir, exist_ok=True)
    x_grid = np.linspace(0.02, 0.98, 100)
    SV = _build_x1_path(n_active, x_grid)
    df = evaluate_along(model, SV)

    ref = None
    if reference_csv and os.path.exists(reference_csv):
        ref = pd.read_csv(reference_csv)
        ref = ref[ref["x"] >= 0.1]

    fig, ax = plt.subplots(1, 4, figsize=(21, 5))
    ax[0].plot(df["x_1"], df["r"], color="C3", lw=2, label="NN")
    if ref is not None: ax[0].plot(ref["x"], ref["r"], color="C0", lw=1, ls="--", label="Cheb")
    ax[0].set_xlabel(r"$x_1$"); ax[0].set_ylabel("r"); ax[0].grid(alpha=.3); ax[0].legend()

    ax[1].plot(df["x_1"], df["pi"], color="C3", lw=2, label="NN")
    if ref is not None: ax[1].plot(ref["x"], ref["pi"], color="C0", lw=1, ls="--", label="Cheb")
    ax[1].set_xlabel(r"$x_1$"); ax[1].set_ylabel(r"$\pi$"); ax[1].grid(alpha=.3); ax[1].legend()

    ax[2].plot(df["x_1"], df["y"], color="C3", lw=2, label="NN")
    if ref is not None: ax[2].plot(ref["x"], ref["y"], color="C0", lw=1, ls="--", label="Cheb")
    ax[2].set_xlabel(r"$x_1$"); ax[2].set_ylabel("y"); ax[2].grid(alpha=.3); ax[2].legend()

    for i in range(n_active):
        ax[3].plot(df["x_1"], df[f"alpha_{i+1}"], lw=2, label=f"NN $\\alpha_{{{i+1}}}$")
    # Chebyshev reference: per-agent risky shares alpha_1, alpha_2, ...
    if ref is not None:
        for i in range(n_active):
            col = f"alpha_{i+1}"
            if col in ref.columns:
                ax[3].plot(ref["x"], ref[col], color="k", lw=1,
                           ls="--" if i == 0 else "-.",
                           label=f"Cheb $\\alpha_{{{i+1}}}$")
    if alpha_caps is not None:
        for i, cap in enumerate(alpha_caps):
            if cap < 1e3:
                ax[3].axhline(cap, color=f"C{i}", lw=0.8, ls=":", alpha=0.7, label=f"cap $\\alpha_{{{i+1}}}={cap}$")
    ax[3].set_xlabel(r"$x_1$"); ax[3].set_ylabel(r"active $\alpha$"); ax[3].grid(alpha=.3); ax[3].legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "equilibrium_symmetric_slice.pdf")
    plt.savefig(out_path, dpi=120)
    plt.close(fig)

    df.to_csv(os.path.join(output_dir, "equilibrium_symmetric_slice.csv"), index=False)
    return out_path


# ---------------------------------------------------------------------------
# Validation-loss & sampling-method comparison helpers
# ---------------------------------------------------------------------------
def _sample_validation_simplex(n_active: int, n_samples: int = 10_000,
                                eps: float = 0.02, max_sum: float = 0.98,
                                seed: int = 0):
    """Draw ``n_samples`` Dirichlet points on the open (N+1)-simplex with
    epsilon truncation — identical distribution to the training sampler so
    every method sees the same evaluation grid."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    alpha = torch.ones(n_active + 1, device="cpu")
    # torch.distributions.Dirichlet doesn't take a generator, so we sample
    # uniformly and renormalise — that gives the same Dirichlet(1, ..., 1).
    u = torch.rand((n_samples, n_active + 1), device="cpu", generator=g)
    samples = -torch.log(u + 1e-30)
    samples = samples / samples.sum(dim=-1, keepdim=True)
    samples = eps + (max_sum - (n_active + 1) * eps) * samples
    return samples[:, :n_active]


def evaluate_validation_losses(model: PDEModel, SV_val: torch.Tensor, chunk_size: int = 1000):
    """Compute per-component loss averages on the validation tensor.

    Returns
    -------
    dict with keys
        "hjb"          : aggregate HJB loss (sum over loss components starting with ``hjbeq``)
        "endog"        : aggregate endogenous-equation loss
        "total"        : sum of all components (HJB + endog + constraint, etc.)
        "components"   : the full mapping {label -> mean loss}
    """
    n_total = SV_val.shape[0]
    n_chunks = (n_total + chunk_size - 1) // chunk_size
    accum: dict = {}
    for c in range(n_chunks):
        SV_chunk = SV_val[c * chunk_size:(c + 1) * chunk_size].clone().to(model.device)
        SV_chunk = _state_for_model(model, SV_chunk)
        SV_chunk.requires_grad_(True)
        for i, sv_name in enumerate(model.state_variables):
            model.variable_val_dict[sv_name] = SV_chunk[:, i:i + 1]
        model.variable_val_dict["SV"] = SV_chunk
        model.update_variables(SV_chunk)
        model.loss_fn()
        for k, v in model.loss_val_dict.items():
            accum[k] = accum.get(k, 0.0) + float(v.detach().cpu().item())
        del SV_chunk
        gc.collect()
        torch.cuda.empty_cache()
    components = {k: v / n_chunks for k, v in accum.items()}
    hjb   = sum(v for k, v in components.items() if k.startswith("hjbeq"))
    endog = sum(v for k, v in components.items() if k.startswith("endogeq"))
    total = sum(components.values())
    return {"hjb": hjb, "endog": endog, "total": total, "components": components}


def compare_losses(models: dict, baseline_key: str = "basic", n_active: int = 2,
                   n_samples: int = 10_000, chunk_size: int = 1000,
                   seed: int = 0,
                   extra_components: dict[str, str] = {"endogeq_vi_active": "Portfolio Choice FOC loss"},
                ):
    """Compare HJB / vi_active / total losses across multiple training methods.

    Parameters
    ----------
    models : dict[str, PDEModel]
        e.g. ``{"basic": model_basic, "RAR": model_rar, "loss_weight": model_lw}``.
        Order is preserved; the resulting DataFrame uses dict-insertion order.
    baseline_key : str
        Method whose losses anchor the 0% improvement column.
    extra_components : tuple[str]
        Extra per-component keys (e.g. ``"endogeq_vi_active"``) to surface as
        their own columns in the table.  These come from ``model.loss_val_dict``
        so they must match the registered equation labels.
    extra_labels : tuple[str]
        Display labels for ``extra_components`` (same length).

    Returns
    -------
    pandas.DataFrame  with absolute-loss columns followed by
    percentage-improvement columns (one of each for HJB, every extra
    component, and Total).  Percentage-improvement is 0 for the baseline
    row and ``100 * (baseline - method) / baseline`` for the others
    (positive means method beats baseline).
    """
    assert baseline_key in models, f"baseline {baseline_key!r} not in models {list(models)}"
    # All methods evaluate on the *same* validation set for a fair comparison.
    SV_val = _sample_validation_simplex(n_active - 1, n_samples=n_samples, seed=seed)

    rows = {}
    for name, mdl in models.items():
        losses = evaluate_validation_losses(mdl, SV_val, chunk_size=chunk_size)
        row = {"HJB loss": losses["hjb"]}
        for key, label in extra_components.items():
            row[label] = float(losses["components"].get(key, float("nan")))
        row["Total loss"] = losses["total"]
        rows[name] = row

    abs_cols = ["HJB loss", *extra_components.values(), "Total loss"]
    base = rows[baseline_key]
    for name, row in rows.items():
        for col in abs_cols:
            base_val = base[col]
            if name == baseline_key:
                row[f"{col} improvement"] = 0.0
            else:
                row[f"{col} improvement"] = 100.0 * (base_val - row[col]) / (abs(base_val) + 1e-30)

    ordered = abs_cols + [f"{c} improvement" for c in abs_cols]
    df = pd.DataFrame.from_dict(rows, orient="index")[ordered]
    return df

def format_sci(x):
    if not np.isfinite(x):
        return "--"
    sci_str = f"{x:.2e}"  # Convert to scientific notation
    base, exp = sci_str.split("e")  # Split into base and exponent
    exp = int(exp)  # Convert exponent to integer to remove leading zeros and '+'
    if exp == 0:
        return f"{base}"
    else:
        return f"${base} \\times 10^{{{exp}}}$"

def format_pct(x):
    if not np.isfinite(x):
        return "--"
    return f"{x:.2f}\\%"

def format_loss_df(loss_df: pd.DataFrame):
    """Format the loss DataFrame for LaTeX export.

    Absolute-loss columns are rendered in scientific notation, percentage-
    improvement columns are rendered as a percentage with two decimals.
    """
    loss_df = loss_df.copy()
    for col in loss_df.columns:
        if "improvement" in col:
            loss_df[col] = loss_df[col].apply(format_pct)
        else:
            loss_df[col] = loss_df[col].apply(format_sci)
    return loss_df.style.to_latex(hrules=True)


def plot_rar_anchors(rar_model_path: str, n_active: int = 2,
                     out_path: str = None, timestepping: bool = False):
    """Scatter-plot of RAR anchor points (meaningful for low-dimensional state)."""
    if timestepping:
        adir = os.path.join(rar_model_path, "anchor_points")
        if not os.path.isdir(adir):
            print(f"[plot_rar_anchors] missing {adir}; skipping")
            return None
        files = sorted(f for f in os.listdir(adir) if f.endswith(".npy"))
        if not files:
            return None
        anchors = np.load(os.path.join(adir, files[-1]))
    else:
        anchor_path = os.path.join(rar_model_path, "model_anchor_points.npy")
        if not os.path.exists(anchor_path):
            print(f"[plot_rar_anchors] no anchor points file at {anchor_path}; "
                  f"was the model trained with RAR sampling?")
            return None
        anchors = np.load(anchor_path)
    if anchors.ndim != 2 or anchors.shape[1] < 1:
        print(f"[plot_rar_anchors] unexpected anchor shape {anchors.shape}.")
        return None
    n_share = n_active - 1
    if anchors.shape[1] == n_share + 1 and timestepping:
        x_col, y_col = 0, -1
        xlab, ylab = "$x_1$", "$t$"
    elif anchors.shape[1] == n_share and n_share == 1:
        x_col, y_col = 0, 0
        xlab, ylab = "$x_1$", "count"
    elif anchors.shape[1] >= 2:
        x_col, y_col = 0, 1
        xlab, ylab = "$x_1$", "$x_2$"
    else:
        print(f"[plot_rar_anchors] cannot plot anchor shape {anchors.shape}.")
        return None

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    if x_col == y_col:
        ax.hist(anchors[:, x_col], bins=30, alpha=0.7)
    else:
        sc = ax.scatter(anchors[:, x_col], anchors[:, y_col], c=np.arange(len(anchors)),
                        cmap="viridis", s=10, alpha=0.7,
                        label=f"RAR anchors (K={len(anchors)})")
        fig.colorbar(sc, ax=ax, label="anchor index (oldest -> newest)")
    ax.set_xlabel(xlab, fontsize=18)
    ax.set_ylabel(ylab, fontsize=18)
    # ax.legend(loc="upper right")
    plt.tight_layout()
    if out_path is None:
        out_path = os.path.join(rar_model_path, "rar_anchors.pdf")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Paper-ready focused plots: y, alpha-sums, and loss-decay panels.
# ---------------------------------------------------------------------------
def _evaluate_methods_on_symmetric_slice(models: dict, n_active: int,
                                         x_min: float = 0.05,
                                         x_max: float = 0.95,
                                         n_points: int = 100):
    """Evaluate every method along the x_1 cut on the {sum_i x_i = 1} simplex:
    x_1 = x_grid, x_2 = ... = x_N = (1 - x_1)/(N-1)."""
    x_grid = np.linspace(x_min, x_max, n_points)
    SV = _build_x1_path(n_active, x_grid)
    method_dfs = {name: evaluate_along(model, SV) for name, model in models.items()}
    return x_grid, method_dfs


def _split_constrained(alpha_caps, n_active: int):
    """Return (unconstrained_indices, constrained_indices) for active agents."""
    if alpha_caps is None:
        return list(range(n_active)), []
    caps = np.asarray(alpha_caps, dtype=float).reshape(-1)
    unc_idx = [i for i in range(n_active) if caps[i] >= 1e3]
    con_idx = [i for i in range(n_active) if caps[i] <  1e3]
    return unc_idx, con_idx


def plot_y_comparison(models: dict, n_active: int, output_dir: str,
                      reference_csv: str = None,
                      file_name: str = "y_comparison.pdf"):
    """Figure 1.  y(x) (weighted average of the value-function multiplier)
    vs the total wealth share ``sum_i x_i``.

    One curve per method (typically ``basic`` and ``RAR``).  The Chebyshev
    numerical solution is overlaid as a dashed black reference when
    ``reference_csv`` exists.  The reference is exact in the symmetric
    2-agent case and an approximation otherwise.
    """
    os.makedirs(output_dir, exist_ok=True)
    _, method_dfs = _evaluate_methods_on_symmetric_slice(models, n_active)

    ref = None
    if reference_csv and os.path.exists(reference_csv):
        ref = pd.read_csv(reference_csv)
        ref = ref[ref["x"] >= 0.05]

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.2))
    colors = plt.get_cmap("tab10")
    for c, (name, df) in enumerate(method_dfs.items()):
        ls, mk = METHOD_PLOT_STYLES[c % len(METHOD_PLOT_STYLES)]
        ax.plot(df["x_1"], df["y"], color=colors(c), lw=2.0, ls=ls,
                marker=mk, markevery=12, markersize=7, label=name)
    if ref is not None and "y" in ref.columns:
        ax.plot(ref["x"], ref["y"], color="k", lw=1.3, ls=":", label="Chebyshev")
    ax.set_xlabel(r"$x_1$", fontsize=18)
    ax.set_ylabel(r"$y(x)$", fontsize=20)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(alpha=.3)
    ax.legend(fontsize=14, frameon=False)
    plt.tight_layout()
    out_path = os.path.join(output_dir, file_name)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_alpha_sums_comparison(models: dict, n_active: int, output_dir: str,
                                reference_csv: str = None,
                                alpha_caps=None,
                                file_name: str = "alpha_sums_comparison.pdf"):
    """Figure 2.  Single-axes overlay of the *sum* of constrained alphas and
    the *sum* of unconstrained alphas, plotted against ``sum_i x_i``.

    Convention:
      * solid lines  -- unconstrained sum  ``sum_{i in U} alpha_i``
      * dashed lines -- constrained sum    ``sum_{i in C} alpha_i``
      * one colour per method; black lines are the Chebyshev reference.

    The numerical reference assumes symmetry across each group (so all
    unconstrained agents share ``alpha_u`` and all constrained agents share
    ``alpha_c``); accordingly we scale the cheb column by the group size.
    This is exact for the 2-agent symmetric cases (``sym2``, ``sym2_const``).
    """
    os.makedirs(output_dir, exist_ok=True)
    _, method_dfs = _evaluate_methods_on_symmetric_slice(models, n_active)
    unc_idx, con_idx = _split_constrained(alpha_caps, n_active)

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.2))
    colors = plt.get_cmap("tab10")
    for c, (name, df) in enumerate(method_dfs.items()):
        _, mk = METHOD_PLOT_STYLES[c % len(METHOD_PLOT_STYLES)]
        if len(unc_idx) > 0:
            sum_unc = sum(df[f"alpha_{i+1}"] for i in unc_idx)
            ax.plot(df["x_1"], sum_unc, color=colors(c), lw=2.0, ls="-",
                    marker=mk, markevery=12, markersize=7, label=f"{name}  $\\sum_{{i\\in U}}\\alpha_i$")
        if len(con_idx) > 0:
            sum_con = sum(df[f"alpha_{i+1}"] for i in con_idx)
            ax.plot(df["x_1"], sum_con, color=colors(c), lw=2.0, ls="--",
                    marker=mk, markevery=12, markersize=7, label=f"{name}  $\\sum_{{i\\in C}}\\alpha_i$")

    if reference_csv and os.path.exists(reference_csv):
        ref = pd.read_csv(reference_csv)
        ref = ref[ref["x"] >= 0.05]
        n_unc, n_con = len(unc_idx), len(con_idx)
        ref_cols = [f"alpha_{i+1}" for i in range(n_active) if f"alpha_{i+1}" in ref.columns]
        if n_unc > 0 and all(f"alpha_{i+1}" in ref.columns for i in unc_idx):
            sum_unc_ref = sum(ref[f"alpha_{i+1}"] for i in unc_idx)
            ax.plot(ref["x"], sum_unc_ref, color="k", lw=1.4, ls="-", label=f"Cheb  $\\sum_{{i\\in U}}\\alpha_i$")
        if n_con > 0 and all(f"alpha_{i+1}" in ref.columns for i in con_idx):
            sum_con_ref = sum(ref[f"alpha_{i+1}"] for i in con_idx)
            ax.plot(ref["x"], sum_con_ref, color="k", lw=1.4, ls="--", label=f"Cheb  $\\sum_{{i\\in C}}\\alpha_i$")

    if alpha_caps is not None:
        caps = np.asarray(alpha_caps, dtype=float).reshape(-1)
        # show the cumulative cap as a horizontal reference for the
        # constrained sum (sum of finite caps)
        finite = caps[caps < 1e3]
        if finite.size > 0:
            ax.axhline(float(finite.sum()), color="gray", lw=0.8, ls=":", alpha=0.7, label=f"$\\sum_{{i\\in C}}$ cap = {float(finite.sum()):.2f}")

    ax.set_xlabel(r"$x_1$", fontsize=18)
    ax.set_ylabel(r"$\sum_{i} \alpha_i$", fontsize=20)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(alpha=.3)
    ax.legend(fontsize=11, frameon=False, ncol=2)
    plt.tight_layout()
    out_path = os.path.join(output_dir, file_name)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_alpha_xi_comparison(models: dict, n_active: int, output_dir: str):
    """Figure 2.  Single-axes overlay of the *sum* of constrained alphas and
    the *sum* of unconstrained alphas, plotted against ``sum_i x_i``.

    Convention:
      * solid lines  -- unconstrained sum  ``sum_{i in U} alpha_i``
      * dashed lines -- constrained sum    ``sum_{i in C} alpha_i``
      * one colour per method; black lines are the Chebyshev reference.

    The numerical reference assumes symmetry across each group (so all
    unconstrained agents share ``alpha_u`` and all constrained agents share
    ``alpha_c``); accordingly we scale the cheb column by the group size.
    This is exact for the 2-agent symmetric cases (``sym2``, ``sym2_const``).
    """
    os.makedirs(output_dir, exist_ok=True)
    _, method_dfs = _evaluate_methods_on_symmetric_slice(models, n_active)
    
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.2))
    colors = plt.get_cmap("tab10")
    for c, (name, df) in enumerate(method_dfs.items()):
        for i, m in [(1, "o"), (n_active // 2, "s"), (n_active, "^")]:
            ax.scatter(df[f"x_{i}"], df[f"alpha_{i}"], color=colors(c), s=20, marker=m, label=f"{name} " + "$\\alpha_{" + str(i) + "}$")

    ax.set_xlabel(r"$x_i$", fontsize=18)
    ax.set_ylabel(r"$\alpha_i$", fontsize=20)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(alpha=.3)
    ax.legend(fontsize=11, frameon=False, ncol=2)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "alpha_comparison.pdf")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.2))
    colors = plt.get_cmap("tab10")
    for c, (name, df) in enumerate(method_dfs.items()):
        for i, m in [(1, "o"), (n_active // 2, "s"), (n_active, "^")]:
            ax.scatter(df[f"x_{i}"], df[f"xi_{i}"], color=colors(c), s=20, marker=m, label=f"{name} " + "$\\xi_{" + str(i) + "}$")

    ax.set_xlabel(r"$x_i$", fontsize=18)
    ax.set_ylabel(r"$\xi_i$", fontsize=20)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(alpha=.3)
    ax.legend(fontsize=11, frameon=False, ncol=2)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "xi_comparison.pdf")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_loss_decay(model_paths: dict, output_dir: str,
                    timestepping_map: dict | None = None,
                    file_name: str = "loss_decay.pdf",
                    loss_csv_name: str = "model_loss.csv",
                    show_running_min: bool = True,
                    targets: tuple = (("hjbeq_1", "HJB"),
                                      ("endogeq_vi_active", "Portfolio Choice FOC"))):
    """Figure 3.  Loss-decay panel for HJB and ``vi_active`` across epochs.

    ``model_paths`` maps a method name to its training directory; each
    directory must contain ``model_loss.csv`` (the per-``loss_log_interval``
    record produced by ``PDEModel.train_model``).

    We deliberately use ``model_loss.csv`` rather than ``model_min_loss.csv``
    because the latter file does not record the *actual* epoch on which each
    new minimum was attained -- its ``epoch`` column is just a running index
    of new-minimum events.  ``model_loss.csv`` carries the true epoch axis;
    overlaying a running-min envelope (``cummin``) yields the monotone
    "best-so-far" curve typically shown in papers, while the faint raw line
    behind it preserves the training dynamics.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(1, len(targets), figsize=(6.2 * len(targets), 4.8),
                           squeeze=False)
    ax = ax[0]
    colors = plt.get_cmap("tab10")
    for c, (name, path) in enumerate(model_paths.items()):
        ts = (timestepping_map or {}).get(name, False)
        csv_name = "model_global_min_loss.csv" if ts else loss_csv_name
        loss_path = os.path.join(path, csv_name)
        if not os.path.exists(loss_path):
            print(f"[plot_loss_decay] missing {loss_path}; skipping {name}")
            continue
        df = pd.read_csv(loss_path)
        # the very first row records the pre-training random-init loss
        # (often >> 1) which compresses the log axis; drop it
        df = df.iloc[1:] if len(df) > 1 else df
        for j, (col, _label) in enumerate(targets):
            if col not in df.columns:
                continue
            if show_running_min:
                ax[j].semilogy(df["epoch"], df[col], color=colors(c), lw=0.9, alpha=0.35)
                rmin = df[col].cummin()
                ax[j].semilogy(df["epoch"], rmin, color=colors(c), lw=2.2, label=f"{name}")
            else:
                ax[j].semilogy(df["epoch"], df[col], color=colors(c), lw=2.0, label=name)

    for j, (_col, label) in enumerate(targets):
        ax[j].set_xlabel("epoch", fontsize=15)
        ax[j].set_ylabel(f"{label} loss", fontsize=15)
        ax[j].tick_params(axis="both", labelsize=12)
        ax[j].legend(fontsize=12, frameon=False)
    plt.tight_layout()
    out_path = os.path.join(output_dir, file_name)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_loss_weights(model_path: str, output_dir: str, file_name: str = "loss_weight.pdf",
                      timestepping: bool = False):
    mapping = {
        "endogeq_mc": "Market clearing",
        "endogeq_vi_active": "Portfolio FOC",
        "endogeq_pricing": "Pricing",
        "hjbeq_1": "HJB"
    }
    if timestepping:
        wdir = os.path.join(model_path, "loss_weight_logs")
        if not os.path.isdir(wdir):
            print(f"[plot_loss_weights] missing {wdir}; skipping")
            return
        files = sorted(f for f in os.listdir(wdir) if f.endswith(".csv"))
        if not files:
            return
        df = pd.read_csv(os.path.join(wdir, files[0]))
    else:
        fpath = os.path.join(model_path, "model_loss_weight.csv")
        if not os.path.exists(fpath):
            print(f"[plot_loss_weights] missing {fpath}; skipping")
            return
        df = pd.read_csv(fpath)
    fig, ax = plt.subplots(1, 1, figsize=(11, 6))
    colors = plt.get_cmap("tab10")
    for i, col in enumerate(mapping):
        ax.plot(df["epoch"], df[col], color=colors(i), label=mapping[col])

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=16, frameon=False)
    ax.tick_params(axis="both", labelsize=14)
    plt.tight_layout()
    out_path = os.path.join(output_dir, file_name)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def compute_consumption_errors(models: dict, n_active: int,
                                reference_csv: str,
                                baseline_key: str = "basic",
                                output_csv: str = None,
                                n_points: int = 200):
    """Quantify how much active learning (RAR) reduces the error in the
    aggregate consumption-wealth ratio ``y(x)`` against the Chebyshev
    numerical reference.

    In this model the aggregate consumption-wealth ratio equals
    ``y = sum_i x_i xi_i`` (active agents only, up to the discount factor ``rho``),
    so the L^p errors of ``y`` are directly errors in aggregate consumption.

    Returns
    -------
    pandas.DataFrame indexed by method with columns
        ``L1, L2, Linf``                  -- absolute errors
        ``rel_L1, rel_L2, rel_Linf``      -- relative errors (denominator: mean |y_ref|, max |y_ref|)
        ``L2 reduction vs <baseline> (%)`` -- positive = method beats baseline
        ``Linf reduction vs <baseline> (%)``
    Evaluated on the symmetric slice ``x_1 = ... = x_N = x_total / N`` for
    ``n_points`` values of x_total on [0.05, 0.95].
    """
    if not (reference_csv and os.path.exists(reference_csv)):
        print(f"[compute_consumption_errors] no reference at {reference_csv}; skipping consumption-error analysis")
        return None
    ref = pd.read_csv(reference_csv)
    ref = ref[ref["x"] >= 0.05]
    x_grid = np.linspace(0.05, 0.95, n_points)
    y_ref = np.interp(x_grid, ref["x"].values, ref["y"].values)
    denom_mean = float(np.mean(np.abs(y_ref)) + 1e-30)
    denom_max  = float(np.max(np.abs(y_ref))  + 1e-30)

    SV = _build_x1_path(n_active, x_grid)
    rows = {}
    for name, model in models.items():
        df = evaluate_along(model, SV)
        err = df["y"].values - y_ref
        L1   = float(np.mean(np.abs(err)))
        L2   = float(np.sqrt(np.mean(err ** 2)))
        Linf = float(np.max(np.abs(err)))
        rows[name] = {
            "L1":       L1,
            "L2":       L2,
            "Linf":     Linf,
            "rel L1":   L1 / denom_mean,
            "rel L2":   L2 / denom_mean,
            "rel Linf": Linf / denom_max,
        }

    if baseline_key in rows:
        base = rows[baseline_key]
        for name, row in rows.items():
            for col in ("L1", "L2", "Linf"):
                base_val = base[col]
                if name == baseline_key:
                    row[f"{col} reduction"] = 0.0
                else:
                    row[f"{col} reduction"] = 100.0 * (base_val - row[col]) / (abs(base_val) + 1e-30)

    df_err = pd.DataFrame.from_dict(rows, orient="index")

    if output_csv is not None:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df_err.to_csv(output_csv)
    return df_err


def format_consumption_errors(df_err: pd.DataFrame):
    """LaTeX-format the consumption-error DataFrame returned by
    ``compute_consumption_errors`` -- scientific notation for the L-errors
    and percentages for the ``... reduction ...`` columns.
    """
    df = df_err.copy()
    for col in df.columns:
        if "reduction" in col:
            df[col] = df[col].apply(format_pct)
        else:
            df[col] = df[col].apply(format_sci)
    return df.style.to_latex(hrules=True)


def _eval_hjb_residual_on_states(model, SV_states: torch.Tensor,
                                  chunk_size: int = 1000):
    """Evaluate the per-sample HJB residual magnitude on arbitrary state
    points without touching the training loop.  Returns numpy arrays
    ``(residual_per_sample, y_per_sample)``.
    """
    n_total = SV_states.shape[0]
    n_chunks = (n_total + chunk_size - 1) // chunk_size
    per_sample_r, per_sample_y = [], []
    for c in range(n_chunks):
        SV_chunk = SV_states[c * chunk_size:(c + 1) * chunk_size].clone().to(model.device)
        SV_chunk = _state_for_model(model, SV_chunk)
        SV_chunk.requires_grad_(True)
        for i, sv_name in enumerate(model.state_variables):
            model.variable_val_dict[sv_name] = SV_chunk[:, i:i + 1]
        model.variable_val_dict["SV"] = SV_chunk
        model.update_variables(SV_chunk)
        # compute_hjb_n returns sum_k hjb_k^2 per sample
        hjb_sq = model.variable_val_dict["hjb"].detach().cpu().numpy().reshape(-1)
        per_sample_r.append(np.sqrt(np.maximum(hjb_sq, 0.0)))
        per_sample_y.append(model.variable_val_dict["y"].detach().cpu().numpy().reshape(-1))
        del SV_chunk
        gc.collect()
        torch.cuda.empty_cache()
    return np.concatenate(per_sample_r), np.concatenate(per_sample_y)


def _residual_stats_table(res_dict: dict, baseline_key: str):
    """Internal: build the standard L^1/L^2/L^inf + percentage-reduction table
    from a ``{name -> (per_sample_residual, per_sample_y)}`` mapping.
    """
    rows = {}
    for name, (r, y) in res_dict.items():
        L1   = float(np.mean(r))
        L2   = float(np.sqrt(np.mean(r ** 2)))
        Linf = float(np.max(r))
        y_bar = float(np.mean(np.abs(y))) + 1e-30
        rows[name] = {
            "L1":       L1,
            "L2":       L2,
            "Linf":     Linf,
            "rel L1":   L1 / y_bar,
            "rel L2":   L2 / y_bar,
            "rel Linf": Linf / y_bar,
        }
    if baseline_key in rows:
        base = rows[baseline_key]
        for name, row in rows.items():
            for col in ("L1", "L2", "Linf"):
                base_val = base[col]
                if name == baseline_key:
                    row[f"{col} reduction"] = 0.0
                else:
                    row[f"{col} reduction"] = 100.0 * (base_val - row[col]) / (abs(base_val) + 1e-30)
    return pd.DataFrame.from_dict(rows, orient="index")


def compute_consumption_errors_residual_on_slice(models: dict, n_active: int,
                                                  baseline_key: str = "basic",
                                                  output_csv: str = None,
                                                  n_points: int = 200,
                                                  x_min: float = 0.05,
                                                  x_max: float = 0.95):
    """Reference-free consumption error on the *symmetric slice*
    :math:`x_1 = \\dots = x_N = x_{\\rm total}/N`.

    This evaluates the HJB residual on **exactly** the same state points
    used by ``compute_consumption_errors`` (the diagonal slice), eliminating
    the domain-mismatch confound that arises when one metric is averaged
    over the full simplex (where RAR puts anchor points in the corners) and
    the other is averaged over the diagonal (which never visits the
    corners).  Because both metrics now sample the same x, their rankings
    are directly comparable.
    """
    x_grid = np.linspace(x_min, x_max, n_points)
    SV_np = _build_x1_path(n_active, x_grid)
    SV_states = torch.tensor(SV_np, device=device, dtype=torch.get_default_dtype())

    res_dict = {name: _eval_hjb_residual_on_states(m, SV_states)
                for name, m in models.items()}
    df = _residual_stats_table(res_dict, baseline_key)

    if output_csv is not None:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df.to_csv(output_csv)
    return df


def compute_consumption_errors_residual(models: dict, n_active: int,
                                         n_samples: int = 10_000,
                                         baseline_key: str = "basic",
                                         output_csv: str = None,
                                         seed: int = 0,
                                         chunk_size: int = 1000):
    """Reference-free measure of "consumption error" via the HJB residual.

    Rationale
    ---------
    The HJB equation is the consumption-savings optimality condition for
    each agent: at the true equilibrium, the per-agent HJB residual is
    identically zero on the entire state simplex.  The :math:`L^p` norm of
    that residual on a dense validation grid is therefore a *model-free*
    proxy for "how wrong consumption is" -- no numerical reference is
    required, only the model itself.  This is the standard PINN "physics
    error" (e.g. Raissi 2017) specialised to our setting.

    For each validation sample :math:`s` on the open N-simplex we form
    :math:`r(s) := \\sqrt{\\sum_{k=1}^{N+1} h_k(s)^2}` -- the L^2 norm across
    agents of the individual HJB residuals -- and report :math:`L^1, L^2,
    L^\\infty` over the validation sample of :math:`r(s)`.  We additionally
    report the same statistics normalised by the average aggregate
    consumption-wealth ratio :math:`\\bar y`, giving a dimensionless
    "relative consumption error".

    Returns
    -------
    pandas.DataFrame indexed by method with columns
        ``L1, L2, Linf``                  -- absolute (residual units)
        ``rel_L1, rel_L2, rel_Linf``      -- divided by mean |y|
        ``L1/L2/Linf reduction vs <baseline> (%)`` -- positive = method beats baseline
    """
    SV_val = _sample_validation_simplex(n_active - 1, n_samples=n_samples, seed=seed)
    res_dict = {name: _eval_hjb_residual_on_states(m, SV_val, chunk_size=chunk_size) for name, m in models.items()}
    df = _residual_stats_table(res_dict, baseline_key)

    if output_csv is not None:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df.to_csv(output_csv)
    return df


# ---------------------------------------------------------------------------
# Welfare-equivalent loss conversion (apples-to-apples table).
#
# The four PDE residuals (HJB, vi_active, mc, pricing) live in
# different units, so the raw MSE table doesn't tell you which loss matters
# economically.  We convert each into a *certainty-equivalent consumption-
# wealth-ratio* (c/W) loss using the appropriate Taylor coefficient:
#
#   HJB residual h_i is *first-order* welfare-relevant (xi_i is c_i/W_i):
#       Delta(c/W) ~ rho * |h|
#
#   Portfolio residuals delta_alpha are *second-order* welfare-relevant
#   (envelope theorem at the FOC):
#       Delta(c/W) ~ 1/2 * gamma * |sigma_R|^2 * (delta_alpha)^2
#
#   Market-clearing residual delta_alpha_tot enters with the wealth-
#   share-weighted gamma_bar(x).
#
#   Pricing residual delta_pi maps via the inverse demand curve
#       delta_alpha_imp = delta_pi / (gamma_bar |sigma_R|^2)
#   then through the Merton coefficient -> 1/(2 gamma_bar |sigma_R|^2)
#   times (delta_pi)^2.
# ---------------------------------------------------------------------------
def compute_welfare_equivalent_losses(models: dict, n_active: int,
                                       n_samples: int = 10_000,
                                       baseline_key: str = "basic",
                                       output_csv: str = None,
                                       seed: int = 0,
                                       chunk_size: int = 1000,
                                       visible_cols: tuple = (
                                           "HJB (c/W)",
                                           "Portfolio Choice FOC (c/W)",
                                           "total (c/W)",
                                       )):
    """Return per-component certainty-equivalent consumption-wealth-ratio
    losses (units of 1/time) for each method, evaluated on a dense Dirichlet
    sample of the simplex.

    All conversions are state-dependent (gamma_bar(x), sigma_R(x) etc.) and
    aggregated by averaging over the validation sample.  The total column
    is the SUM of *all* per-component welfare losses (HJB + Portfolio
    Choice FOC active + market-clearing + pricing), since welfare cost is
    additive across equilibrium errors to leading order.

    Note: every component is computed internally; ``visible_cols`` controls
    which columns are exposed in the returned table (and the CSV / LaTeX
    file).  Default surfaces only the headline trio ``HJB``, ``Portfolio
    Choice FOC``, ``total``.  Pass a longer tuple (e.g. include
    ``"mc (c/W)"``) to surface diagnostics.

    Returns
    -------
    pandas.DataFrame indexed by method with the absolute c/W columns in
    ``visible_cols`` followed by their percentage-reduction-vs-baseline
    counterparts (positive means the method beats the baseline).
    """
    SV_val = _sample_validation_simplex(n_active - 1, n_samples=n_samples, seed=seed)

    rows: dict = {}
    for name, model in models.items():
        n_chunks = (n_samples + chunk_size - 1) // chunk_size
        sum_hjb_we   = 0.0
        sum_vi_we    = np.zeros(n_active, dtype=np.float64)
        sum_mc_we    = 0.0
        sum_pricing_we = 0.0
        n_total = 0
        for c in range(n_chunks):
            SV_chunk = SV_val[c * chunk_size:(c + 1) * chunk_size].clone().to(model.device)
            SV_chunk = _state_for_model(model, SV_chunk)
            B = SV_chunk.shape[0]
            SV_chunk.requires_grad_(True)
            for i, sv_name in enumerate(model.state_variables):
                model.variable_val_dict[sv_name] = SV_chunk[:, i:i + 1]
            model.variable_val_dict["SV"] = SV_chunk
            model.update_variables(SV_chunk)
            vd = model.variable_val_dict

            # --- parameters / state-dependent helpers -----------------------
            ga = vd["gamma_active"].reshape(1, -1)        # (1, N)
            pa = vd["psi_active"].reshape(1, -1)
            rho_v = vd["rho"]                              # python float OR tensor
            n_share = getattr(model, "n_share", SV_chunk.shape[-1] - (1 if "t" in model.state_variables else 0))
            x_active = compute_x_full(SV_chunk[:, :n_share])            # (B, N) incl. dependent x_N
            alpha_a  = vd["alpha_active"]                  # (B, N)
            sigR     = vd["sigR"]                          # (B, 2)
            sigR_sq  = (sigR ** 2).sum(dim=-1, keepdim=True)   # (B, 1)
            sigR_norm = vd["sigR_norm"]                    # (B, 1)
            # state-dependent average risk aversion: gamma_bar(x)
            gamma_bar = (x_active * ga).sum(dim=-1, keepdim=True) # (B, 1)

            # --- per-component residuals ------------------------------------
            xi_active   = vd["xi_active"]
            muxi_active = vd["muxi_active"]
            sigxi_active = vd["sigxi_active"]
            r_v         = vd["r"]
            eta_v       = vd["eta"]

            hjb_active = (
                rho_v * pa
                + (1 - pa) * (r_v + eta_v * alpha_a * sigR_norm
                              - ga / 2 * (alpha_a * sigR_norm) ** 2)
                + muxi_active
                + (1 - ga) * (sigxi_active * sigR.unsqueeze(1)).sum(dim=-1) * alpha_a
                + 0.5 * (pa - ga) / (1 - pa + 1e-8) * (sigxi_active ** 2).sum(dim=-1)
                - xi_active
            ) / rho_v                               # (B, 1)

            mc_resid = (x_active * alpha_a).sum(dim=-1, keepdim=True) - 1.0                             # (B, 1)
            alpha_caps_v = vd["alpha_caps"]              # (1, N)
            foc_a_target = vd["foc_active_target"]
            vi_resid = torch.minimum(alpha_caps_v - alpha_a, foc_a_target - alpha_a)   # (B, N)
            pricing_resid = vd["pi"] - vd["pricing_target"]      # (B, 1)

            # --- welfare-equivalent contributions ---------------------------
            # HJB: linear conversion -> rho * |h|
            sum_hjb_we  += float((rho_v * hjb_active.abs()).sum().item())
            # vi_active per agent: 1/2 gamma_i sigR^2 (delta_alpha_i)^2
            vi_we_per_b = 0.5 * ga * sigR_sq * vi_resid ** 2          # (B, N)
            sum_vi_we += vi_we_per_b.sum(dim=0).detach().cpu().numpy()
            # mc: 1/2 gamma_bar(x) sigR^2 (delta_alpha_tot)^2
            sum_mc_we     += float((0.5 * gamma_bar * sigR_sq * mc_resid ** 2).sum().item())
            # pricing: 1/(2 gamma_bar(x) sigR^2) (delta_pi)^2
            sum_pricing_we += float((pricing_resid ** 2 / (2.0 * gamma_bar * sigR_sq + 1e-30)).sum().item())
            n_total += B
            del SV_chunk
            gc.collect()
            torch.cuda.empty_cache()

        mean_hjb_we    = sum_hjb_we    / n_total
        mean_vi_we     = sum_vi_we     / n_total          # (N,) per agent
        mean_mc_we     = sum_mc_we     / n_total
        mean_pricing_we = sum_pricing_we / n_total

        total_hjb = mean_hjb_we
        total_vi  = float(mean_vi_we.sum())
        total = total_hjb + total_vi + mean_mc_we + mean_pricing_we

        rows[name] = {
            "HJB (c/W)":                    total_hjb,
            "Portfolio Choice FOC (c/W)":   total_vi,
            "mc (c/W)":                     mean_mc_we,
            "pricing (c/W)":                mean_pricing_we,
            "total (c/W)":                  total,
        }
        # per-agent Portfolio Choice FOC breakdown (useful for asymmetric cases)
        for i in range(n_active):
            rows[name][f"Portfolio Choice FOC_{i+1} (c/W)"] = float(mean_vi_we[i])

    # always compute reductions for the full set of absolute columns; the
    # ``visible_cols`` filter below decides what gets surfaced.
    all_abs_cols = ["HJB (c/W)", "Portfolio Choice FOC (c/W)", "mc (c/W)", "pricing (c/W)", "total (c/W)"]
    if baseline_key in rows:
        base = rows[baseline_key]
        for name, row in rows.items():
            for col in all_abs_cols:
                base_val = base[col]
                if name == baseline_key:
                    row[f"{col} reduction"] = 0.0
                else:
                    row[f"{col} reduction"] = 100.0 * (base_val - row[col]) / (abs(base_val) + 1e-30)

    # surface only the requested visible columns + their reduction counterparts
    visible_cols = list(visible_cols)
    ordered = list(visible_cols)
    ordered += [f"{c} reduction" for c in visible_cols]
    df = pd.DataFrame.from_dict(rows, orient="index")[ordered]

    if output_csv is not None:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df.to_csv(output_csv)
    return df


def format_welfare_equivalent_losses(df: pd.DataFrame):
    """LaTeX format the welfare-equivalent loss table -- scientific for the
    absolute c/W columns and percentage for the reduction columns."""
    out = df.copy()
    for col in out.columns:
        if "reduction" in col:
            out[col] = out[col].apply(format_pct)
        else:
            out[col] = out[col].apply(format_sci)
    return out.style.to_latex(hrules=True)


def plot_method_comparison(models: dict, n_active: int, output_dir: str,
                           reference_csv: str = None,
                           alpha_caps=None, file_name: str = "method_comparison.pdf"):
    """Overlay (r, pi, y, q, active alphas) across training methods.

    Evaluated on the symmetric slice ``x_1 = ... = x_N = x_total / N``.
    Any number of methods is supported (extensible to e.g. a ``loss_weight``
    method); each is drawn in its own colour with a numerical reference (if
    the CSV exists) overlaid as a dashed black line.

    The active-alpha panel stacks every method's alpha_1, ..., alpha_N on a
    single axis (one *colour* per method, one *linestyle* per agent index).
    """
    os.makedirs(output_dir, exist_ok=True)
    x_grid = np.linspace(0.05, 0.95, 100)
    SV = _build_x1_path(n_active, x_grid)
    method_dfs = {name: evaluate_along(model, SV) for name, model in models.items()}

    ref = None
    if reference_csv and os.path.exists(reference_csv):
        ref = pd.read_csv(reference_csv)
        ref = ref[ref["x"] >= 0.1]

    colors = plt.get_cmap("tab10")
    line_styles = ["-", "--", "-.", ":"]
    # one line per method (scalar quantities, plus a combined active-alpha panel)
    scalar_quantities = [
        ("r",            "$r$"),
        ("pi",           "$\\pi$"),
        ("y",            "$y$"),
        ("q",            "$q$"),
    ]
    ref_keys = {"r": "r", "pi": "pi", "y": "y", "q": None}
    n_panels = len(scalar_quantities) + 1  # +1 for combined active-alpha panel

    fig, ax = plt.subplots(1, n_panels, figsize=(8.2 * n_panels, 8))
    ax = ax.flatten()

    # ---- standard scalar panels ------------------------------------------
    for j, (key, label) in enumerate(scalar_quantities):
        for c, (name, df) in enumerate(method_dfs.items()):
            ls, mk = METHOD_PLOT_STYLES[c % len(METHOD_PLOT_STYLES)]
            ax[j].plot(df["x_1"], df[key], color=colors(c), lw=2, ls=ls,
                       marker=mk, markevery=12, markersize=7, label=name)
        rk = ref_keys.get(key)
        if ref is not None and rk is not None and rk in ref.columns:
            ax[j].plot(ref["x"], ref[rk], color="k", ls=":", lw=1, label="Cheb")
        ax[j].set_xlabel(r"$x_1$"); ax[j].set_ylabel(label, fontsize=20)
        ax[j].grid(alpha=.3); ax[j].legend(fontsize=16, frameon=False)

    # ---- combined active-alpha panel -------------------------------------
    j = len(scalar_quantities)
    for c, (name, df) in enumerate(method_dfs.items()):
        _, mk = METHOD_PLOT_STYLES[c % len(METHOD_PLOT_STYLES)]
        for i in range(n_active):
            ls = line_styles[i % len(line_styles)]
            ax[j].plot(df["x_1"], df[f"alpha_{i+1}"], color=colors(c), ls=ls, lw=2,
                       marker=mk, markevery=12, markersize=6, label=f"{name} $\\alpha_{{{i+1}}}$")
    # Chebyshev reference: per-agent risky shares alpha_1, alpha_2, ...
    if ref is not None:
        for i in range(n_active):
            col = f"alpha_{i+1}"
            if col in ref.columns:
                ax[j].plot(ref["x"], ref[col], color="k", ls="--" if i == 0 else "-.", lw=1.4, label=f"Cheb $\\alpha_{{{i+1}}}$")
    if alpha_caps is not None:
        for i, cap in enumerate(alpha_caps):
            if cap < 1e3:
                ax[j].axhline(cap, color=f"C{i}", lw=0.8, ls=":", alpha=0.7, label=f"cap $\\alpha_{{{i+1}}}={cap}$")
    ax[j].set_xlabel(r"$x_1$"); ax[j].set_ylabel(r"active $\alpha_i$", fontsize=20)
    ax[j].grid(alpha=.3); ax[j].legend(fontsize=16, ncol=2, frameon=False)

    for a in ax:
        a.tick_params(axis='both', labelsize=20)
    plt.tight_layout()
    out_path = os.path.join(output_dir, file_name)
    plt.savefig(out_path)
    plt.close(fig)
    return out_path

# ---------------------------------------------------------------------------
# Parameter helper
# ---------------------------------------------------------------------------
def make_params(gamma_active_vec, psi_active_vec=None, omega_active_vec=None,
                rho=0.05, mu=0.0183, sigma_vec=(0.0357, 0.0),
                kappa=0.0):
    n = len(gamma_active_vec)
    if psi_active_vec is None:
        psi_active_vec = [1.5] * n
    if omega_active_vec is None:
        omega_active_vec = [0.5 / n] * n
    return {
        "gamma_active":  torch.tensor(gamma_active_vec, device=device, dtype=torch.get_default_dtype()),
        "psi_active":    torch.tensor(psi_active_vec,   device=device, dtype=torch.get_default_dtype()),
        "omega_active":  torch.tensor(omega_active_vec, device=device, dtype=torch.get_default_dtype()),
        "rho":           rho,
        "mu":            mu,
        "sigma":         torch.tensor(list(sigma_vec), device=device, dtype=torch.get_default_dtype()),
        "kappa":         kappa,
    }


# ---------------------------------------------------------------------------
# The 8 training configurations
# ---------------------------------------------------------------------------
# name -> (timestepping, rar, loss_balancing)
CONFIGS = {
    "basic":            (False, False, False),
    "basic_rar":        (False, True,  False),
    "basic_lb":         (False, False, True),
    "basic_rar_lb":     (False, True,  True),
    "timestep":         (True,  False, False),
    "timestep_rar":     (True,  True,  False),
    "timestep_lb":      (True,  False, True),
    "timestep_rar_lb":  (True,  True,  True),
}


# Per-method plot styling: each plotted method gets a distinct (linestyle,
# marker) pair so basic vs. the best method are separable in black & white.
METHOD_PLOT_STYLES = [
    ("-",  "o"),
    ("--", "s"),
    ("-.", "^"),
    (":",  "D"),
]


def select_plot_methods(models: dict, loss_df=None, welfare_df=None,
                        baseline_key: str = "basic",
                        val_improvement_col: str = "Total loss improvement",
                        welfare_improvement_col: str = "total (c/W) reduction"):
    """Return ``{name: model}`` containing the baseline plus the method(s) with
    the biggest improvement over the baseline -- by validation total loss and
    by total welfare-equivalent loss.  The two criteria may pick the same
    method (then only one is added).  Order: baseline first, then the picks.
    """
    ordered = [baseline_key] if baseline_key in models else list(models)[:1]
    for df, col in [(loss_df, val_improvement_col),
                    (welfare_df, welfare_improvement_col)]:
        if df is None or col not in df.columns:
            continue
        cand = df.drop(index=baseline_key, errors="ignore")[col].dropna()
        cand = cand[np.isfinite(cand)]
        if len(cand) == 0:
            continue
        best = cand.idxmax()
        if best in models and best not in ordered:
            ordered.append(best)
    return {k: models[k] for k in ordered if k in models}


# ---------------------------------------------------------------------------
# Entry point — validation runs
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    print(sys.argv)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["sym2", "sym2_const", "asym2", "asym2_const", "n20", "mix5", "mix20", "mix50"],
                        default="sym2")
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--outer", type=int, default=50, help="num_outer_iterations for time-stepping configs")
    parser.add_argument("--max_t", type=float, default=1,
                        help="pseudo-time slab width for time-stepping (smaller = gentler march; >0.5 can NaN)")
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--float64", action="store_true")
    args = parser.parse_args()
    print(args)

    if args.float64:
        torch.set_default_dtype(torch.float64)
        base_dir = "./models/GP_NN_NAgents_faster_64bits"
    else:
        torch.set_default_dtype(torch.float32)
        base_dir = "./models/GP_NN_NAgents_faster"

    alpha_caps = None  # default: every active agent unconstrained
    if args.case == "sym2":
        n = 2
        gamma_active = [4.0, 4.0]
        psi_active = [1.5, 1.5]
        suffix = "sym2"
    elif args.case == "asym2":
        n = 2
        gamma_active = [8.0, 4.0]
        psi_active = [1.5, 1.5]
        suffix = "asym2"
    elif args.case == "sym2_const":
        n = 2
        gamma_active = [4.0, 4.0]
        psi_active = [1.5, 1.5]
        alpha_caps = [1e6, 1.8]
        suffix = "sym2_const"
    elif args.case == "asym2_const":
        n = 2
        gamma_active = [8.0, 4.0]
        psi_active = [1.5, 1.5]
        alpha_caps = [1e6, 1.8]
        suffix = "asym2_const"
    elif args.case == "mix5":
        n = 4
        gamma_active = [5.0, 4.0, 3.0, 2.0]
        psi_active = [1.5] * 4
        alpha_caps = [1e6, 1e6, 1e6, 1.8]
        suffix = "mix5"
    elif args.case == "mix20":
        n = 20
        gamma_active = [21 - i for i in range(20)]
        psi_active = [1.5] * 20
        alpha_caps = [1e6] * 19 + [1.8]
        suffix = "mix20"
    elif args.case == "mix50":
        n = 50
        gamma_active = [51 - i for i in range(50)]
        psi_active = [1.5] * 50
        alpha_caps = [1e6] * 49 + [1.8]
        suffix = "mix50"
    print(f"[gp_n_agents_NN] case={args.case}  N={n}  gamma_active={gamma_active} alpha_caps={alpha_caps}")

    case_dir = os.path.join(base_dir, suffix)
    params = make_params(gamma_active, psi_active_vec=psi_active)

    # Time-stepping warm start: the stationary value multipliers xi_i are O(rho)
    # (the consumption-wealth ratio), NOT the library default of 1.0.  Starting
    # the march ~20x too high (xi=1) forces many outer loops just to relax down,
    # with a large transient d(xi)/dt inflating the HJB the whole way.  Seed the
    # boundary at xi ~ rho and alpha ~ 1 (market-clearing baseline) so the march
    # begins near equilibrium.
    rho_val = params["rho"]
    ts_init_guess = {f"xi_{i+1}": float(rho_val) for i in range(n)}
    ts_init_guess.update({f"alpha_{i+1}": 1.0 for i in range(n)})

    models, model_paths, ts_map = {}, {}, {}
    for name in CONFIGS.keys():
        ts, rar, lb = CONFIGS[name]
        mpath = os.path.join(case_dir, name)
        print(f"\n{('=== ' + name + ' ==='):=^80}")
        models[name] = get_model(
            mpath, n, params,
            model_size=[args.width] * args.layers,
            n_epochs=args.epochs, batch_size=args.batch, lr=args.lr,
            alpha_caps=alpha_caps,
            timestepping=ts, rar=rar, loss_balancing=lb,
            num_outer=args.outer, num_inner=5000, max_t=args.max_t,
            init_guess=ts_init_guess if ts else None,
        )
        model_paths[name] = mpath
        ts_map[name] = ts
        gc.collect()
        torch.cuda.empty_cache()

    comparison_dir = f"{case_dir}_comparison"
    os.makedirs(comparison_dir, exist_ok=True)
    reference_csv = f"./models/gp_numerical_cheb/{suffix}/cheb_solution.csv"

    for name, ts, rar, lb in [(n, *CONFIGS[n]) for n in models]:
        if lb:
            plot_loss_weights(model_paths[name], comparison_dir,
                              file_name=f"loss_weight_{name}.pdf", timestepping=ts)

    for name, ts, rar, lb in [(n, *CONFIGS[n]) for n in models]:
        if rar:
            plot_rar_anchors(model_paths[name], n_active=n, timestepping=ts,
                             out_path=os.path.join(comparison_dir, f"rar_anchors_{name}.pdf"))

    # ---- loss tables (computed over ALL trained methods) --------------------
    loss_df = compare_losses(models, baseline_key="basic", n_active=n, n_samples=10_000, chunk_size=200)
    loss_csv = os.path.join(comparison_dir, "validation_losses.csv")
    loss_df.to_csv(loss_csv)
    with open(f"{comparison_dir}/validation_losses.tex", "w") as f:
        f.write(format_loss_df(loss_df))
    print("[gp_n_agents_NN] validation loss table:")
    print(loss_df.to_string(float_format=lambda x: f"{x:.4e}"))

    welfare_df = compute_welfare_equivalent_losses(
        models, n_active=n, baseline_key="basic",
        output_csv=os.path.join(comparison_dir, "welfare_equivalent_losses.csv"),
        chunk_size=200,
    )
    torch.cuda.empty_cache()
    gc.collect()
    with open(f"{comparison_dir}/welfare_equivalent_losses.tex", "w") as f:
        f.write(format_welfare_equivalent_losses(welfare_df))

    # ---- pick basic + the best-improving method(s) for the overlay plots ----
    plot_models = select_plot_methods(models, loss_df=loss_df, welfare_df=welfare_df,
                                       baseline_key="basic")
    plot_paths = {k: model_paths[k] for k in plot_models}
    plot_ts = {k: ts_map[k] for k in plot_models}
    print(f"[gp_n_agents_NN] plotting methods: {list(plot_models)}")

    # Overlay plots: only the baseline and the best method, evaluated on the
    # t=0 slice via ``evaluate_along`` and overlaid against the Chebyshev
    # stationary reference.  Each method gets a distinct linestyle + marker.
    y_png = plot_y_comparison(
        plot_models, n_active=n, output_dir=comparison_dir,
        reference_csv=reference_csv, file_name="y_comparison.pdf",
    )
    print(f"[gp_n_agents_NN] y comparison saved to {y_png}")

    alpha_sums_png = plot_alpha_sums_comparison(
        plot_models, n_active=n, output_dir=comparison_dir,
        reference_csv=reference_csv, alpha_caps=alpha_caps,
        file_name="alpha_sums_comparison.pdf",
    )
    print(f"[gp_n_agents_NN] alpha-sums comparison saved to {alpha_sums_png}")

    plot_alpha_xi_comparison(plot_models, n_active=n, output_dir=comparison_dir)

    loss_png = plot_loss_decay(plot_paths, comparison_dir, plot_ts)
    print(f"[gp_n_agents_NN] loss-decay plot saved to {loss_png}")

    cmp_png = plot_method_comparison(
        plot_models, n_active=n, output_dir=comparison_dir,
        reference_csv=reference_csv, alpha_caps=alpha_caps,
        file_name="method_comparison.pdf",
    )
    print(f"[gp_n_agents_NN] method comparison saved to {cmp_png}")

    # ---- consumption-error tables (computed over ALL trained methods) -------
    err_df_resid_slice = compute_consumption_errors_residual_on_slice(
        models, n_active=n, baseline_key="basic",
        output_csv=os.path.join(comparison_dir, "consumption_errors_residual_slice.csv"),
    )
    torch.cuda.empty_cache()
    gc.collect()
    with open(f"{comparison_dir}/consumption_errors_residual_slice.tex", "w") as f:
        f.write(format_consumption_errors(err_df_resid_slice))
    print("[gp_n_agents_NN] reference-free consumption-error on symmetric slice:")

    err_df_resid = compute_consumption_errors_residual(
        models, n_active=n, baseline_key="basic",
        output_csv=os.path.join(comparison_dir, "consumption_errors_residual.csv"),
        chunk_size=200,
    )
    torch.cuda.empty_cache()
    gc.collect()
    with open(f"{comparison_dir}/consumption_errors_residual.tex", "w") as f:
        f.write(format_consumption_errors(err_df_resid))

    err_df = compute_consumption_errors(
        models, n_active=n, reference_csv=reference_csv, baseline_key="basic",
        output_csv=os.path.join(comparison_dir, "consumption_errors.csv"),
    )
    if err_df is not None:
        with open(f"{comparison_dir}/consumption_errors.tex", "w") as f:
            f.write(format_consumption_errors(err_df))

    print(f"[gp_n_agents_NN] all comparison outputs in {comparison_dir}")

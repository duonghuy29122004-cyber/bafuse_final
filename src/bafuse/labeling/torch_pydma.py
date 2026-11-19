from __future__ import annotations

from contextlib import contextmanager
from functools import partial
import math
from collections.abc import Iterator
from typing import Any

import numpy as np
from scipy.optimize import OptimizeResult
import torch

from pydma.core.objectives import combined_objective
from pydma.electrodes.blend import BlendElectrode
from pydma.utils.roi import build_roi_mask


class TorchOCVEvaluator:
    """Batched CUDA evaluator for PyDMA's non-blended OCV-only objective."""

    def __init__(self, objective: Any, device: torch.device) -> None:
        if not isinstance(objective, partial) or objective.func is not combined_objective:
            raise ValueError("CUDA fitting only supports PyDMA's combined_objective partial.")

        options = objective.keywords or {}
        if options.get("w_ocv", 0.0) <= 0 or options.get("w_dva", 0.0) != 0 or options.get("w_ica", 0.0) != 0:
            raise ValueError("CUDA fitting requires OCV-only objective weights.")
        if options.get("anode_is_blend") or options.get("cathode_is_blend"):
            raise ValueError("CUDA fitting does not support blend electrodes.")
        if options.get("inhom_anode_offset", 0.0) != 0 or options.get("inhom_cathode_offset", 0.0) != 0:
            raise ValueError("CUDA fitting requires zero electrode inhomogeneity offsets.")

        anode = options["anode"]
        cathode = options["cathode"]
        if isinstance(anode, BlendElectrode) or isinstance(cathode, BlendElectrode):
            raise ValueError("CUDA fitting does not support blend electrodes.")

        self.device = device
        self.weight = float(options["w_ocv"])
        self.r_offset_sign = float(options.get("r_offset_sign", 1.0))
        self.current = float(options.get("pocv_current_a", 0.0))
        self.q = torch.as_tensor(options["q"], dtype=torch.float64, device=device).flatten()
        self.measured = torch.as_tensor(
            options["meas_voltage"], dtype=torch.float64, device=device
        ).flatten()
        self.mask = torch.as_tensor(
            build_roi_mask(
                options["q"],
                options["roi_ocv_min"],
                options["roi_ocv_max"],
            ),
            dtype=torch.bool,
            device=device,
        )
        if not torch.any(self.mask):
            raise ValueError("The OCV ROI must include at least one point.")

        self.anode_soc, self.anode_voltage = self._electrode_tensors(anode.soc, anode.voltage)
        self.cathode_soc, self.cathode_voltage = self._electrode_tensors(
            cathode.soc, cathode.voltage
        )

    def _electrode_tensors(
        self,
        soc: np.ndarray,
        voltage: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        order = np.argsort(np.asarray(soc, dtype=np.float64))
        sorted_soc = np.asarray(soc, dtype=np.float64)[order]
        sorted_voltage = np.asarray(voltage, dtype=np.float64)[order]
        unique_soc, first_indices = np.unique(sorted_soc, return_index=True)
        unique_voltage = sorted_voltage[first_indices]
        if unique_soc.size < 2:
            raise ValueError("Each electrode needs at least two unique SOC points.")
        return (
            torch.as_tensor(unique_soc, dtype=torch.float64, device=self.device),
            torch.as_tensor(unique_voltage, dtype=torch.float64, device=self.device),
        )

    def _interpolate(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        soc: torch.Tensor,
        voltage: torch.Tensor,
    ) -> torch.Tensor:
        if torch.any(alpha <= 0):
            raise ValueError("CUDA interpolation requires positive SOC scale parameters.")
        transformed_soc = alpha[:, None] * soc[None, :] + beta[:, None]
        queries = self.q.expand(alpha.shape[0], -1).contiguous()
        indices = torch.searchsorted(transformed_soc.contiguous(), queries)
        lower = (indices - 1).clamp(0, soc.numel() - 2)
        upper = lower + 1
        x0 = torch.gather(transformed_soc, 1, lower)
        x1 = torch.gather(transformed_soc, 1, upper)
        y0 = voltage[lower]
        y1 = voltage[upper]
        fraction = (queries - x0) / (x1 - x0)
        values = y0 + fraction * (y1 - y0)
        in_range = (queries >= transformed_soc[:, :1]) & (
            queries <= transformed_soc[:, -1:]
        )
        return torch.where(in_range, values, torch.zeros_like(values))

    def evaluate(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return weighted objective cost and unweighted OCV RMSE per candidate."""
        params = torch.as_tensor(params, dtype=torch.float64, device=self.device)
        if params.ndim != 2 or params.shape[1] < 9:
            raise ValueError("Candidate parameters must have shape (batch, at least 9).")
        if torch.any(torch.abs(params[:, 6:8]) > 1e-12):
            raise ValueError("CUDA fitting currently requires zero inhomogeneity parameters.")

        anode = self._interpolate(
            params[:, 0], params[:, 1], self.anode_soc, self.anode_voltage
        )
        cathode = self._interpolate(
            params[:, 2], params[:, 3], self.cathode_soc, self.cathode_voltage
        )
        calculated = cathode - anode
        calculated = calculated + (
            self.r_offset_sign * params[:, 8:9] * self.current
        )
        squared_error = torch.square(calculated[:, self.mask] - self.measured[self.mask])
        mse = squared_error.mean(dim=1)
        return self.weight * mse, torch.sqrt(mse)


def cuda_differential_evolution(
    objective: Any,
    bounds: list[tuple[float, float]],
    *,
    callback: Any = None,
    **options: Any,
) -> OptimizeResult:
    """Run batched best/1/bin differential evolution with CUDA objective evaluation."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA differential evolution requires an available CUDA device.")
    if options.get("constraints") or options.get("integrality"):
        raise ValueError("CUDA differential evolution does not support constraints or integrality.")
    if options.get("strategy", "best1bin") != "best1bin":
        raise ValueError("CUDA differential evolution only supports best1bin strategy.")
    if int(options.get("workers", 1)) != 1:
        raise ValueError("CUDA differential evolution manages batched work on one device.")

    device = torch.device("cuda:0")
    evaluator = TorchOCVEvaluator(objective, device)
    bound_array = np.asarray(bounds, dtype=np.float64)
    if bound_array.ndim != 2 or bound_array.shape[1] != 2:
        raise ValueError("Bounds must be a sequence of lower/upper pairs.")
    lower = torch.as_tensor(bound_array[:, 0], dtype=torch.float64, device=device)
    upper = torch.as_tensor(bound_array[:, 1], dtype=torch.float64, device=device)
    free = upper > lower
    free_indices = torch.nonzero(free, as_tuple=True)[0]
    if free_indices.numel() < 2:
        raise ValueError("CUDA differential evolution needs at least two free parameters.")
    if bound_array.shape[0] != 9:
        raise ValueError("CUDA PyDMA currently expects its 9-slot parameter layout.")
    if (
        not torch.isfinite(lower).all()
        or not torch.isfinite(upper).all()
        or torch.any(upper[free] <= lower[free])
    ):
        raise ValueError("CUDA search requires finite, ordered parameter bounds.")

    seed = options.get("seed")
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(int(seed))

    dimension = int(free_indices.numel())
    population_size = int(options.get("popsize", 15)) * dimension
    if population_size < 5 * dimension:
        raise ValueError("Population size must be at least five candidates per free parameter.")
    max_iterations = int(options.get("maxiter", 1000))
    tolerance = float(options.get("tol", 0.01))
    absolute_tolerance = float(options.get("atol", 0.0))
    recombination = float(options.get("recombination", 0.7))
    mutation = options.get("mutation", (0.5, 1.0))
    if isinstance(mutation, (tuple, list)):
        mutation_low, mutation_high = map(float, mutation)
    else:
        mutation_low = mutation_high = float(mutation)
    if not 0 <= recombination <= 1 or not 0 <= mutation_low <= mutation_high <= 2:
        raise ValueError("Invalid recombination or mutation settings.")

    unit = torch.empty((population_size, dimension), dtype=torch.float64, device=device)
    for column in range(dimension):
        strata = (torch.arange(population_size, device=device, dtype=torch.float64) +
                  torch.rand(population_size, dtype=torch.float64, device=device, generator=generator))
        unit[:, column] = strata[torch.randperm(population_size, device=device, generator=generator)] / population_size

    initial = options.get("init")
    if isinstance(initial, np.ndarray):
        if initial.shape != (population_size, len(bounds)):
            raise ValueError("Explicit initial population has an incompatible shape.")
        initial_tensor = torch.as_tensor(initial, dtype=torch.float64, device=device)
        unit = ((initial_tensor[:, free_indices] - lower[free_indices]) /
                (upper[free_indices] - lower[free_indices])).clamp(0.0, 1.0)
    elif initial not in (None, "latinhypercube"):
        raise ValueError("CUDA differential evolution supports Latin-hypercube initialization only.")

    x0 = options.get("x0")
    if x0 is not None:
        x0_tensor = torch.as_tensor(x0, dtype=torch.float64, device=device)
        if x0_tensor.numel() != len(bounds):
            raise ValueError("x0 must contain one value per PyDMA parameter slot.")
        unit[0] = ((x0_tensor[free_indices] - lower[free_indices]) /
                   (upper[free_indices] - lower[free_indices])).clamp(0.0, 1.0)

    def expand(normalized: torch.Tensor) -> torch.Tensor:
        candidates = lower.expand(normalized.shape[0], -1).clone()
        candidates[:, free_indices] = (
            lower[free_indices]
            + normalized * (upper[free_indices] - lower[free_indices])
        )
        return candidates

    def evaluate(normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        costs, rmses = evaluator.evaluate(expand(normalized))
        if not torch.isfinite(costs).all() or not torch.isfinite(rmses).all():
            raise FloatingPointError("CUDA objective produced non-finite values.")
        return costs, rmses

    energies, _ = evaluate(unit)
    function_evaluations = population_size
    iterations = 0
    converged = False
    for iteration in range(max_iterations):
        best = unit[torch.argmin(energies)]
        if mutation_low == mutation_high:
            scale = mutation_low
        else:
            scale = float(
                torch.empty((), dtype=torch.float64, device=device).uniform_(
                    mutation_low, mutation_high, generator=generator
                )
            )

        index_rows = []
        for candidate_index in range(population_size):
            choices = torch.cat(
                (
                    torch.arange(candidate_index, device=device),
                    torch.arange(candidate_index + 1, population_size, device=device),
                )
            )
            sampled = choices[
                torch.randperm(choices.numel(), device=device, generator=generator)[:2]
            ]
            index_rows.append(sampled)
        indices = torch.stack(index_rows)
        mutant = best + scale * (unit[indices[:, 0]] - unit[indices[:, 1]])
        crossover = torch.rand(
            (population_size, dimension), dtype=torch.float64, device=device, generator=generator
        ) < recombination
        forced = torch.randint(dimension, (population_size,), device=device, generator=generator)
        crossover[torch.arange(population_size, device=device), forced] = True
        trial = torch.where(crossover, mutant, unit)
        outside = (trial < 0.0) | (trial > 1.0)
        replacement = torch.rand(
            trial.shape, dtype=torch.float64, device=device, generator=generator
        )
        trial = torch.where(outside, replacement, trial)

        trial_energies, _ = evaluate(trial)
        function_evaluations += population_size
        improved = trial_energies <= energies
        unit = torch.where(improved[:, None], trial, unit)
        energies = torch.where(improved, trial_energies, energies)
        iterations = iteration + 1

        mean_energy = float(energies.mean())
        spread = float(energies.std(unbiased=False))
        convergence = math.inf if mean_energy == 0 else spread / abs(mean_energy)
        if callback is not None and callback(
            expand(unit[torch.argmin(energies)][None, :])[0].detach().cpu().numpy(),
            convergence,
        ):
            break
        if spread <= absolute_tolerance + tolerance * abs(mean_energy):
            converged = True
            break

    best_unit = unit[torch.argmin(energies)].detach().clone()
    best_cost = float(energies.min())
    polish = bool(options.get("polish", False))
    if polish:
        lower_free = lower[free_indices]
        width_free = upper[free_indices] - lower_free
        start = ((best_unit.clamp(1e-8, 1 - 1e-8).logit()).detach()).requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [start], max_iter=100, tolerance_grad=1e-9, tolerance_change=1e-12
        )
        polish_evaluations = 0

        def closure() -> torch.Tensor:
            nonlocal polish_evaluations
            optimizer.zero_grad()
            candidate = torch.sigmoid(start)
            cost, _ = evaluator.evaluate(expand(candidate[None, :]))
            loss = cost[0]
            loss.backward()
            polish_evaluations += 1
            return loss

        optimizer.step(closure)
        polished = torch.sigmoid(start).detach()
        polished_costs, _ = evaluate(polished[None, :])
        function_evaluations += polish_evaluations + 1
        if float(polished_costs[0]) < best_cost:
            best_unit = polished
            best_cost = float(polished_costs[0])

    best_params = expand(best_unit[None, :])[0].detach().cpu().numpy()
    return OptimizeResult(
        x=best_params,
        fun=best_cost,
        nit=iterations,
        nfev=function_evaluations,
        success=converged,
        message="CUDA best1bin differential evolution converged" if converged else "CUDA differential evolution reached its iteration limit",
    )


@contextmanager
def use_cuda_optimizer() -> Iterator[None]:
    """Temporarily route PyDMA's optimizer calls through the CUDA backend."""
    if not torch.cuda.is_available():
        raise RuntimeError("PyDMA CUDA backend requested, but CUDA is unavailable.")
    import pydma.core.optimizer as optimizer_module

    original = optimizer_module.differential_evolution
    optimizer_module.differential_evolution = cuda_differential_evolution
    try:
        yield
    finally:
        optimizer_module.differential_evolution = original
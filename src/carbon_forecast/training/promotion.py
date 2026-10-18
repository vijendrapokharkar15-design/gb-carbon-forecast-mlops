"""Champion/challenger decision rule.

A pure function: metrics in, decision out. No files, no models,
so every branch of the rule is easy to test.
"""
from dataclasses import dataclass


@dataclass
class Decision:
    promote: bool
    reason: str


def decide(
    challenger: dict,
    baseline: dict,
    champion: dict | None,
    mae_tolerance: float,
    regret_tolerance: float,
) -> Decision:
    """Each metrics dict must contain 'mae' and 'regret_mean', scored on the same window.

    1. Guardrail: the challenger must beat the naive baseline on both metrics.
    2. First model: passing the guardrail is enough.
    3. No-regression gate: promote unless the challenger is worse than the
       champion by more than the tolerance on either metric. A fresher model
       that performs as well is preferred, because the grid drifts.
    """
    c_mae, c_reg = challenger["mae"], challenger["regret_mean"]

    if c_mae >= baseline["mae"] or c_reg >= baseline["regret_mean"]:
        return Decision(
            False,
            f"guardrail: challenger (MAE {c_mae:.2f}, regret {c_reg:.2f}) does not beat "
            f"naive (MAE {baseline['mae']:.2f}, regret {baseline['regret_mean']:.2f})",
        )

    if champion is None:
        return Decision(True, "first model: passes the naive guardrail")

    mae_diff = c_mae - champion["mae"]
    reg_diff = c_reg - champion["regret_mean"]

    if mae_diff > mae_tolerance:
        return Decision(False, f"regression: MAE worse than champion by {mae_diff:.2f} (tolerance {mae_tolerance})")
    if reg_diff > regret_tolerance:
        return Decision(False, f"regression: regret worse than champion by {reg_diff:.2f} (tolerance {regret_tolerance})")

    return Decision(True, f"no regression vs champion: MAE {mae_diff:+.2f}, regret {reg_diff:+.2f}")
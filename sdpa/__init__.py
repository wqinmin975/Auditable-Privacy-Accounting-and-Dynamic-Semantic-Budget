"""DP-aware semantic communication: calibration, accounting, budget allocation.

Modules
-------
calibrate : per-component latent statistics, sensitivity, utility weights, residual fraction.
allocate  : closed-form water-filling allocator + RDP / GDP / basic-composition accounting
            + Fano-certified error floor.
train     : reference PyTorch skeleton (true-DP noise + secret invertible shaper T +
            conditioned deprotector).
figures   : IEEE column-width figures and PGFPlots data sources.
"""

from .calibrate import (  # noqa: F401
    Calibration,
    synthetic_calibration,
    reconstruction_calibration,
    task_decoupled_calibration,
    calibrate_from_latents,
)
from .measured import measured_calibration, measure_part_stats  # noqa: F401
from .allocate import (  # noqa: F401
    Allocation,
    waterfill_sigma,
    solve_mi_budget,
    solve_rdp_budget,
    solve_gdp_budget,
    solve_basic_budget,
    static_index_allocation,
    i_max_bits,
    tau_ceiling,
    fano_floor,
    gdp_epsilon,
    rdp_epsilon_delta,
    evaluate,
)

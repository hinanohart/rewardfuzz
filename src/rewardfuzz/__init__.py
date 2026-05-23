"""rewardfuzz — preventive, a-priori auditor for reward / fitness / verifier functions.

Public API (stable surface for 0.1.x):

    from rewardfuzz import audit, AuditReport
    report = audit(my_reward_fn, adapter="callable")
    print(report.hackability)

See :func:`rewardfuzz.audit.audit` for the full signature.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .adapters import list_adapters, load_adapter, register_adapter
from .audit import audit
from .report.schema import AuditReport
from .strategies import list_strategies, register_strategy
from .types import Candidate, RewardResult, Verdict

__all__ = [
    "__version__",
    "audit",
    "AuditReport",
    "RewardResult",
    "Candidate",
    "Verdict",
    "load_adapter",
    "register_adapter",
    "list_adapters",
    "register_strategy",
    "list_strategies",
]

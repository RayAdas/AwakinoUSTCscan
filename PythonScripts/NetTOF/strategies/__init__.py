"""Available network strategies for NetTOF."""

from .regression_strategy import RegressionNetStrategy

STRATEGY_REGISTRY = {
    "regression": RegressionNetStrategy,
}

__all__ = ["STRATEGY_REGISTRY", "RegressionNetStrategy"]

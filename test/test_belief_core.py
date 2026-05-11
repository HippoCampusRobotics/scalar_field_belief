import numpy as np

from scalar_field_belief.belief import ScalarFieldBelief
from scalar_field_belief.config import BeliefConfig


def test_simple_fit_and_query():
    cfg = BeliefConfig(training_iter=5, refit_policy="every_measurement")
    belief = ScalarFieldBelief(cfg)
    belief.add_measurement(0.2, 0.5, 1.0)
    assert belief.has_model()
    mean, var = belief.query(np.array([[0.2, 0.5], [1.0, 2.0]], dtype=float))
    assert mean.shape == (2,)
    assert var.shape == (2,)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(var))
    assert np.all(var >= 0.0)

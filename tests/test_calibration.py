import numpy as np

from datscan.training.calibrate import apply_temperature, fit_temperature


def test_temperature_scaling_round_trip():
    logits = np.array([-2.0, -0.4, 0.7, 2.0])
    targets = np.array([0.0, 0.0, 1.0, 1.0])
    temperature = fit_temperature(logits, targets)
    assert temperature > 0
    probabilities = apply_temperature(logits, temperature)
    assert np.isfinite(probabilities).all()
    assert ((probabilities > 0) & (probabilities < 1)).all()


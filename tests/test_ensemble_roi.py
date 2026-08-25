import numpy as np

from datscan.training.ensemble import blend_probabilities, grid_search_two_model_weights, prediction_diversity


def test_oof_grid_weights_sum_to_one_and_probabilities_are_bounded():
    target = np.array([0, 0, 1, 1], dtype=float)
    global_probability = np.array([0.1, 0.4, 0.6, 0.9])
    roi_probability = np.array([0.2, 0.3, 0.8, 0.7])
    weight, weights, _ = grid_search_two_model_weights(target, global_probability, roi_probability)
    prediction = blend_probabilities(global_probability, roi_probability, weight)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all((prediction > 0) & (prediction < 1))
    diversity = prediction_diversity(target, global_probability, roi_probability)
    assert diversity["classification_disagreement_count"] >= 0

"""Small submission validators."""

import numpy as np
import pandas as pd


def validate_submission(output: pd.DataFrame, template: pd.DataFrame) -> None:
    if list(output.columns) != ["uid", "is_pathologic"]:
        raise ValueError("Submission columns must be exactly uid,is_pathologic")
    if len(output) != len(template) or output["uid"].astype(str).tolist() != template["uid"].astype(str).tolist():
        raise ValueError("Submission ordering does not match submission_format.csv")
    probabilities = output["is_pathologic"].to_numpy(dtype=float)
    if output["uid"].duplicated().any() or not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Invalid submission UID or probability")


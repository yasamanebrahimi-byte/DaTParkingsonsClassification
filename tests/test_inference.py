import numpy as np
import nibabel as nib
import pandas as pd

from datscan.inference.predict import validate_submission


def test_submission_validation():
    template = pd.DataFrame({"uid": ["a", "b"], "is_pathologic": [0.0, 0.0]})
    output = pd.DataFrame({"uid": ["a", "b"], "is_pathologic": [0.2, 0.8]})
    validate_submission(output, template)


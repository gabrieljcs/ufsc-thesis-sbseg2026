from __future__ import annotations

import unittest

from thesis_eval.analysis.glmm import MAIN_PREDICTORS, _prepare_data, _z


class GlmmPreparationTests(unittest.TestCase):
    def test_prepare_data_excludes_reference_and_standardizes_complete_rows(self) -> None:
        rows = [
            {
                "model": "weak_model",
                "model_alignment_pole": "weak",
                "prompt_id": "p1",
                "attack_language": "eng",
                "unsafe_binary": 0,
                "distance": 0.0,
                "spec_score": 1.0,
                "excluded": False,
            },
            {
                "model": "strong_model",
                "model_alignment_pole": "strong",
                "prompt_id": "p2",
                "attack_language": "ita",
                "unsafe_binary": 1,
                "distance": 1.0,
                "spec_score": 3.0,
                "excluded": False,
            },
            {
                "model": "reference_model",
                "model_alignment_pole": "reference",
                "prompt_id": "p3",
                "attack_language": "eng",
                "unsafe_binary": 0,
                "distance": 0.5,
                "spec_score": 2.0,
                "excluded": False,
            },
            {
                "model": "excluded_model",
                "model_alignment_pole": "weak",
                "prompt_id": "p4",
                "attack_language": "eng",
                "unsafe_binary": 0,
                "distance": 0.5,
                "spec_score": 2.0,
                "excluded": True,
            },
        ]

        data = _prepare_data(rows, MAIN_PREDICTORS)

        self.assertEqual(len(data), 2)
        self.assertEqual(set(data["model"]), {"weak_model", "strong_model"})
        self.assertAlmostEqual(float(data["distance_z"].mean()), 0.0)
        self.assertAlmostEqual(float(data["spec_z"].mean()), 0.0)
        self.assertAlmostEqual(float(data["distance_z"].std(ddof=0)), 1.0)

    def test_z_maps_constant_predictor_to_zero(self) -> None:
        import pandas as pd

        standardized = _z(pd.Series([2.0, 2.0, 2.0]))
        self.assertEqual(standardized.tolist(), [0.0, 0.0, 0.0])

    def test_prepare_data_supports_postreview_exclusions(self) -> None:
        base = {
            "model": "weak_model",
            "model_alignment_pole": "weak",
            "unsafe_binary": 0,
            "distance": 0.5,
            "spec_score": 1.0,
            "excluded": False,
            "aligned_language": "por",
            "translation_blaser_status": "pass",
        }
        rows = [
            {**base, "prompt_id": "p1", "attack_language": "por", "distance": 0.0},
            {**base, "prompt_id": "p2", "attack_language": "fin"},
            {
                **base,
                "prompt_id": "p3",
                "attack_language": "eng",
                "translation_blaser_status": "flag",
            },
            {**base, "prompt_id": "p4", "attack_language": "ita"},
        ]

        without_aligned = _prepare_data(rows, MAIN_PREDICTORS, exclude_aligned=True)
        without_caveats = _prepare_data(
            rows,
            MAIN_PREDICTORS,
            exclude_attack_languages=frozenset({"fin", "swa", "ukr"}),
        )
        without_flags = _prepare_data(rows, MAIN_PREDICTORS, exclude_blaser_flagged=True)

        self.assertEqual(set(without_aligned["prompt_id"]), {"p2", "p3", "p4"})
        self.assertEqual(set(without_caveats["prompt_id"]), {"p1", "p3", "p4"})
        self.assertEqual(set(without_flags["prompt_id"]), {"p1", "p2", "p4"})


if __name__ == "__main__":
    unittest.main()

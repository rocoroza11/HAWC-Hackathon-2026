from pathlib import Path

import joblib

from train_sklearn_pue_stage1 import (
    DATA_PATH,
    print_stage1_report,
    train_stage1_forecaster,
)

from train_sklearn_pue_stage2 import (
    print_stage2_report,
    train_stage2_pue,
)

from stage3 import (
    DATA_INTERVAL_HOURS,
    build_stage3_impact_frame,
    print_stage3_report,
)


MODEL_DIR = Path("models")
STAGE1_MODEL_PATH = MODEL_DIR / "stage1_model.joblib"
STAGE2_MODEL_PATH = MODEL_DIR / "stage2_model.joblib"


def _extract_model(result, stage_name):
    """
    Extract the fitted sklearn model/pipeline from a training result.

    Supports the common result shapes used by the training modules:
      - an object with `.model`, `.pipeline`, `.forecaster`, or `.estimator`
      - a dict containing one of those keys
      - the fitted estimator itself (anything exposing `predict`)
    """
    candidate_names = ("model", "pipeline", "forecaster", "estimator")

    for name in candidate_names:
        if isinstance(result, dict) and name in result:
            candidate = result[name]
            if hasattr(candidate, "predict"):
                return candidate

        candidate = getattr(result, name, None)
        if candidate is not None and hasattr(candidate, "predict"):
            return candidate

    if hasattr(result, "predict"):
        return result

    raise TypeError(
        f"Could not find a fitted {stage_name} model in the training result. "
        "Expose the fitted sklearn model as `.model` (or `.pipeline`, "
        "`.forecaster`, `.estimator`) or return the estimator directly."
    )


def main(data_path=DATA_PATH):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Stage 1: ML model
    # -------------------------
    stage1_result = train_stage1_forecaster(data_path)
    print_stage1_report(stage1_result)

    stage1_model = _extract_model(stage1_result, "Stage 1")
    joblib.dump(stage1_model, STAGE1_MODEL_PATH)
    print(f"Saved Stage 1 model to: {STAGE1_MODEL_PATH}")

    # -------------------------
    # Stage 2: ML model
    # -------------------------
    stage2_model, eval_df, train, test, predictions = train_stage2_pue(
        stage1_result
    )
    print_stage2_report(eval_df, train, test, predictions)

    # Stage 2's first return value is the fitted model/pipeline.
    if not hasattr(stage2_model, "predict"):
        stage2_model = _extract_model(stage2_model, "Stage 2")

    joblib.dump(stage2_model, STAGE2_MODEL_PATH)
    print(f"Saved Stage 2 model to: {STAGE2_MODEL_PATH}")

    # -------------------------
    # Stage 3: business layer
    # -------------------------
    stage3_frame = build_stage3_impact_frame(
        test=test,
        stage2_predictions=predictions,
        # No IT-load telemetry is available yet, so report per MW of IT load.
        it_load_kw=None,
        interval_hours=DATA_INTERVAL_HOURS,
    )

    print_stage3_report(stage3_frame)


if __name__ == "__main__":
    main()

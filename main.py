import joblib 

from train_sklearn_pue_stage1 import (
	DATA_PATH,
	print_stage1_report,
	train_stage1_forecaster,
)

from train_sklearn_pue_stage2 import (
    print_stage2_report, 
    train_stage2_pue
)

from stage3 import (
	DATA_INTERVAL_HOURS,
	build_stage3_impact_frame,
	print_stage3_report
)

def main(data_path=DATA_PATH):
	stage1_result = train_stage1_forecaster(data_path)
	print_stage1_report(stage1_result)

	_, eval_df, train, test, predictions = train_stage2_pue(stage1_result)
	print_stage2_report(eval_df, train, test, predictions)

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
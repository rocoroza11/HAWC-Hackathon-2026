from train_sklearn_pue_stage1 import (
	DATA_PATH,
	print_stage1_report,
	train_stage1_forecaster,
)
from train_sklearn_pue_stage2 import print_stage2_report, train_stage2_pue


def main(data_path=DATA_PATH):
	stage1_result = train_stage1_forecaster(data_path)
	stage2_result = train_stage2_pue(stage1_result)

	print_stage1_report(stage1_result)
	print_stage2_report(*stage2_result[1:])
	return stage1_result, stage2_result


if __name__ == "__main__":
	main()
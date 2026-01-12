from DataPreprocessing import PreprocessingConfig, DataPreprocessing
from Logger import Logger

if __name__ == '__main__':
    log = Logger("Main")

    log.info("Start processing data...")
    dp = DataPreprocessing(PreprocessingConfig())

    dp.load_datasets("./csv")
    dp.merge_dataset()
    dp.feature_preparation()
    dp.anomaly_detection()
    dp.output_result("./save")

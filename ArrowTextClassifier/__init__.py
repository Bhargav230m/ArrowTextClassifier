from main import Model

PACKAGE_NAME = "ARROW_TEXT_CLASSIFIER"
FUTURE = "NEW DATASETS AND MODEL COMING SOON!!"

def example_dataset_format():
    return '{"label":"normal","example":"Hey there!"}, {"label":"normal","example":"Hi!"}, {"label":"toxic","example":"You suck!"} --> Convert this to parquet'
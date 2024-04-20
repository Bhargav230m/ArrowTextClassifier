from ArrowTextClassifier.summarize_model import summarize_model
from ArrowTextClassifier.classify_text import classifyText
from ArrowTextClassifier.train import train


class Model:
    def __init__(self, name):
        self.name = name

    def summarize(
        self,
        model_path,
        hyperparams_path,
        device="cpu",
        vocabulary_path=None,
        modelSummary_write_path=None,
    ):
        return summarize_model(
            model_path=model_path,
            hparams_path=hyperparams_path,
            vocab_path=vocabulary_path,
            device=device,
            modelSummary_write_path=modelSummary_write_path,
        )

    def classify(
        self,
        model_path,
        hyperparams_path,
        text,
        vocabulary_path,
        device="cpu",
        log=True
    ):
        return classifyText(
            vocab_file=vocabulary_path,
            model_path=model_path,
            hyperparametersPath=hyperparams_path,
            text=text,
            device=device,
            log=log
        )

    def train_model(
        self,
        dataset,
        EMBEDDING_DIM=100,
        NUM_FILTERS=100,
        FILTER_SIZES=[3, 4, 5],
        DROPOUT=0.2,
        BATCH_SIZE=32,
        NUM_EPOCHS=10,
        WEIGHT_DECAY=0.001,
        LEARNING_RATE=0.0001,
        device="cpu",
    ):
        return train(
            dataset=dataset,
            EMBEDDING_DIM=EMBEDDING_DIM,
            NUM_FILTERS=NUM_FILTERS,
            FILTER_SIZES=FILTER_SIZES,
            DROPOUT=DROPOUT,
            BATCH_SIZE=BATCH_SIZE,
            NUM_EPOCHS=NUM_EPOCHS,
            WEIGHT_DECAY=WEIGHT_DECAY,
            LEARNING_RATE=LEARNING_RATE,
            name=self.name,
            device=device,
        )

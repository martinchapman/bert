import random, torch, transformers
import numpy as np


class Util:

    # https://docs.pytorch.org/docs/stable/notes/randomness.html
    @staticmethod
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        if transformers.file_utils.is_torch_available():
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def class_balance(dataset, labels):
        # Create dictionary holding label counts
        counts = {label: sum(dataset[label]) for label in labels}
        # Map class to count and percentage prevalence
        return {
            label: (count, count / len(dataset) * 100)
            for label, count in counts.items()
        }

    # https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
    @staticmethod
    def reformat_predictions(predictions, threshold):
        probs = torch.sigmoid(torch.Tensor(predictions))
        reformatted = np.zeros(probs.shape)
        reformatted[np.where(probs >= threshold)] = 1
        return reformatted

# from transformers.models.deberta import

from transformers import AutoTokenizer, AutoModel
from transformers.models.deberta import DebertaModel


if __name__ == "__main__":

    dt = AutoTokenizer.from_pretrained("microsoft/deberta-large")

    tokenization_out = dt("The quick brown fox niente", return_tensors="pt")

    dm = AutoModel.from_pretrained("microsoft/deberta-large")

    dm(**tokenization_out)

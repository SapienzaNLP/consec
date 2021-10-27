import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.consec_dataset import build_samples_generator_from_disambiguation_corpus, ConsecDataset
from src.consec_tokenizer import ConsecTokenizer, DeBERTaTokenizer
from src.dependency_finder import EmptyDependencyFinder, PPMIPolysemyDependencyFinder
from src.disambiguation_corpora import WordNetCorpus
from src.sense_inventories import WordNetSenseInventory, XlWSDSenseInventory


if __name__ == "__main__":

    pl.seed_everything(seed=96)

    tokenizer = DeBERTaTokenizer(
        transformer_model="microsoft/deberta-large",
        # target_marker=("{{{", "}}}"),
        target_marker=("<d>", "</d>"),
        context_definitions_token="CONTEXT_DEFS",
        context_markers=dict(number=1, pattern=("DEF_SEP", "DEF_END")),
        # context_markers=dict(number=100, pattern=("<c#I#>", "</c#I#>")),
        add_prefix_space=True,
    )

    wordnet = WordNetSenseInventory("data/WSD_Evaluation_Framework/Data_Validation/candidatesWN30.txt")

    disambiguation_corpus = WordNetCorpus(
        # "data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor",
        # "data/xl-wsd/evaluation_datasets/dev-it/dev-it",
        "data/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL",
        materialize=False,
        cached=False,
    )

    dependency_finder = PPMIPolysemyDependencyFinder(
        sense_inventory=wordnet,
        single_counter_path="data/pmi/lemma_counter.txt",
        pair_counter_path="data/pmi/pairs_counter.txt",
        energy=0.7,
        minimum_ppmi=0.1,
        max_dependencies=9,
        with_pos=False,
    )

    generate_samples = build_samples_generator_from_disambiguation_corpus(
        sense_inventory=wordnet,
        # disambiguation_corpus=[disambiguation_corpus, disambiguation_corpus],
        disambiguation_corpus=disambiguation_corpus,
        dependency_finder=dependency_finder,
        sentence_window=2,
        randomize_sentence_window=True,
        remove_multilabel_instances=True,
        shuffle_definitions=True,
        randomize_dependencies=False,
        sense_frequencies_path="data/sense-counts/semcor_sense_counts.tsv",
    )

    consec_dataset = ConsecDataset(
        samples_generator=generate_samples,
        tokenizer=tokenizer,
        use_definition_start=True,
        # text_encoding_strategy="positional",
        text_encoding_strategy="relative-positions",
        tokens_per_batch=1536,
        max_batch_size=128,
        section_size=10,
        prebatch=True,
        shuffle=True,
        max_length=tokenizer.model_max_length,
    )

    depends_from_counts = []

    data_loader = DataLoader(consec_dataset, None, num_workers=0)

    with open("/tmp/consec.log", "w") as f:
        for dataset_sample in data_loader:
            input_ids_len = dataset_sample["input_ids"].shape[-1]
            if input_ids_len > 2000:
                print(tokenizer.tokenizer.batch_decode(dataset_sample["input_ids"]))

    print("Dependencies avg num: {}".format(sum(depends_from_counts) / len(depends_from_counts)))
    print("Dependencies max num: {}".format(max(depends_from_counts)))

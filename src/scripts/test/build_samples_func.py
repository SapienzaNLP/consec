from src.consec_dataset import build_samples_generator_from_disambiguation_corpus
from src.consec_tokenizer import ConsecTokenizer, DeBERTaTokenizer
from src.dependency_finder import EmptyDependencyFinder, PPMIPolysemyDependencyFinder
from src.disambiguation_corpora import WordNetCorpus
from src.sense_inventories import WordNetSenseInventory

if __name__ == "__main__":

    tokenizer = DeBERTaTokenizer(
        transformer_model="microsoft/deberta-large",
        begin_of_mark="{{{",
        end_of_mark="}}}",
        definition_sep_token="DEF_SEP",
        definition_end_token="DEF_END",
        context_definitions_token="CONTEXT_DEFS",
        add_prefix_space=True,
    )

    wordnet = WordNetSenseInventory("data/WSD_Evaluation_Framework/Data_Validation/candidatesWN30.txt")

    semeval2007 = WordNetCorpus(
        "data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007", materialize=False, cached=False
    )

    # dependency_finder = EmptyDependencyFinder()

    dependency_finder = PPMIPolysemyDependencyFinder(
        sense_inventory=wordnet,
        single_counter_path="data/pmi/lemma_counter.txt",
        pair_counter_path="data/pmi/pairs_counter.txt",
        energy=0.7,
    )

    generate_samples = build_samples_generator_from_disambiguation_corpus(
        tokenizer=tokenizer,
        sense_inventory=wordnet,
        disambiguation_corpus=semeval2007,
        dependency_finder=dependency_finder,
        sentence_window=2,
        randomize_sentence_window=True,
        remove_multilabel_instances=True,
        shuffle_definitions=True,
        randomize_dependencies=True,
        max_dependencies=-1,
    )

    for sample in generate_samples():
        print(sample)

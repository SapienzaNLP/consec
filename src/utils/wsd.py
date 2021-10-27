import tempfile
import xml.etree.cElementTree as ET
from typing import NamedTuple, Optional, List, Callable, Tuple, Iterable
from xml.dom import minidom

from src.utils.commons import execute_bash_command

pos_map = {
    # U-POS
    "NOUN": "n",
    "VERB": "v",
    "ADJ": "a",
    "ADV": "r",
    "PROPN": "n",
    # PEN
    "AFX": "a",
    "JJ": "a",
    "JJR": "a",
    "JJS": "a",
    "MD": "v",
    "NN": "n",
    "NNP": "n",
    "NNPS": "n",
    "NNS": "n",
    "RB": "r",
    "RP": "r",
    "RBR": "r",
    "RBS": "r",
    "VB": "v",
    "VBD": "v",
    "VBG": "v",
    "VBN": "v",
    "VBP": "v",
    "VBZ": "v",
    "WRB": "r",
}


class AnnotatedToken(NamedTuple):
    idx: int
    text: str
    pos: Optional[str] = None
    lemma: Optional[str] = None


class WSDInstance(NamedTuple):
    annotated_token: AnnotatedToken
    labels: Optional[List[str]]
    instance_id: Optional[str]


def read_from_raganato(
    xml_path: str,
    key_path: Optional[str] = None,
    instance_transform: Optional[Callable[[WSDInstance], WSDInstance]] = None,
) -> Iterable[Tuple[str, str, List[WSDInstance]]]:
    def read_by_text_iter(xml_path: str):

        it = ET.iterparse(xml_path, events=("start", "end"))
        _, root = next(it)

        for event, elem in it:
            if event == "end" and elem.tag == "text":
                document_id = elem.attrib["id"]
                for sentence in elem:
                    sentence_id = sentence.attrib["id"]
                    for word in sentence:
                        yield document_id, sentence_id, word

            root.clear()

    mapping = {}

    if key_path is not None:
        with open(key_path) as f:
            for line in f:
                line = line.strip()
                wsd_instance, *labels = line.split(" ")
                mapping[wsd_instance] = labels

    last_seen_document_id = None
    last_seen_sentence_id = None

    for document_id, sentence_id, element in read_by_text_iter(xml_path):

        if last_seen_sentence_id != sentence_id:

            if last_seen_sentence_id is not None:
                yield last_seen_document_id, last_seen_sentence_id, sentence

            sentence = []
            last_seen_document_id = document_id
            last_seen_sentence_id = sentence_id

        annotated_token = AnnotatedToken(
            idx=len(sentence),
            text=element.text,
            pos=element.attrib.get("pos", None),
            lemma=element.attrib.get("lemma", None),
        )

        wsd_instance = WSDInstance(
            annotated_token=annotated_token,
            labels=None
            if element.tag == "wf" or element.attrib["id"] not in mapping
            else mapping[element.attrib["id"]],
            instance_id=None if element.tag == "wf" else element.attrib["id"],
        )

        if instance_transform is not None:
            wsd_instance = instance_transform(wsd_instance)

        sentence.append(wsd_instance)

    yield last_seen_document_id, last_seen_sentence_id, sentence


def expand_raganato_path(path: str) -> Tuple[str, str]:
    return f"{path}.data.xml", f"{path}.gold.key.txt"


class RaganatoBuilder:
    def __init__(self, lang: Optional[str] = None, source: Optional[str] = None):
        self.corpus = ET.Element("corpus")
        self.current_text_section = None
        self.current_sentence_section = None
        self.gold_senses = []

        if lang is not None:
            self.corpus.set("lang", lang)

        if source is not None:
            self.corpus.set("source", source)

    def open_text_section(self, text_id: str, text_source: str = None):
        text_section = ET.SubElement(self.corpus, "text")
        text_section.set("id", text_id)
        if text_source is not None:
            text_section.set("source", text_source)
        self.current_text_section = text_section

    def open_sentence_section(self, sentence_id: str, update_id: bool = True):
        sentence_section = ET.SubElement(self.current_text_section, "sentence")
        if update_id:
            sentence_id = self.compute_id([self.current_text_section.attrib["id"], sentence_id])
        sentence_section.set("id", sentence_id)
        self.current_sentence_section = sentence_section

    def add_annotated_token(
        self,
        token: str,
        lemma: str,
        pos: str,
        instance_id: Optional[str] = None,
        labels: Optional[List[str]] = None,
        update_id: bool = False,
    ):
        if instance_id is not None:
            token_element = ET.SubElement(self.current_sentence_section, "instance")
            if update_id:
                instance_id = self.compute_id([self.current_sentence_section.attrib["id"], instance_id])
            token_element.set("id", instance_id)
            if labels is not None:
                self.gold_senses.append((instance_id, " ".join(labels)))
        else:
            token_element = ET.SubElement(self.current_sentence_section, "wf")
        token_element.set("lemma", lemma)
        token_element.set("pos", pos)
        token_element.text = token

    @staticmethod
    def compute_id(chain_ids: List[str]) -> str:
        return ".".join(chain_ids)

    def store(self, data_output_path: str, labels_output_path: str):
        self.__store_xml(data_output_path)
        self.__store_labels(labels_output_path)

    def __store_xml(self, output_path: str):
        corpus_writer = ET.ElementTree(self.corpus)
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(f"{tmp_dir}/tmp.xml", "wb") as f_xml:
                corpus_writer.write(f_xml, encoding="UTF-8", xml_declaration=True)
            execute_bash_command(f" xmllint --format {tmp_dir}/tmp.xml > {output_path}")

    def __store_labels(self, output_path: str):
        with open(output_path, "w") as f_labels:
            for gold_sense in self.gold_senses:
                f_labels.write(" ".join(gold_sense))
                f_labels.write("\n")

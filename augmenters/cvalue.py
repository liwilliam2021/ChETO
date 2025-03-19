"""
Inspired by the structure and approach of the value paper for Chilean Spanish
and with a different use-case. Here, we merely detect Chilean Spanish and its rule.
Paper: https://arxiv.org/pdf/2204.03031

Requires Python 3.12 for spaCy compatibility.
"""

import spacy
from spacy.matcher import Matcher
import re
import json

CHILENISMO_PATH = "../data/chilenismos_dictionary.json"
QUEISMO_TRIGGERS_PATH = "../data/queísmo_triggers.json"

"""
Rules for detecting specific Chilean Spanish dialect features.
Does NOT modify the input text. Does not mantain state.
"""
class ChileanDialectRules:
    def __init__(self):
        # Load spaCy
        self.nlp = spacy.load("es_core_news_lg")
        self.reflexive_pronouns = {"me", "te", "se", "nos", "os"}

        # Load Chilean Spanish dictionary
        with open(CHILENISMO_PATH, "r", encoding="utf-8") as f:
            self.chilenismos_dictionary = json.load(f)
        # Load example words that could trigger queísmo
        with open(QUEISMO_TRIGGERS_PATH, "r", encoding="utf-8") as f:
            self.queismo_triggers = json.load(f)
        
        # queísmo triggers that are reflexive verbs
        self.reflexive_trigger_words = {
            item[:-2] for item in self.queismo_triggers if item.endswith(
                ("arse", "erse", "irse")
            )
        }
        self.queismo_matcher = Matcher(self.nlp.vocab)
        # Add both adjective + "que" and verb + "que" patterns
        # Adjective pattern
        adj_pattern = [
            {"LEMMA": {"IN": list(self.queismo_triggers)}},
            {"LOWER": "que"}
        ]
        self.queismo_matcher.add("QUEISMO_PATTERN", [adj_pattern])
        # Pattern for reflexive verb + que, e.g., "me aseguré que"
        reflexive_verb_pattern = [
            {"LOWER": {"IN": ["me", "te", "se", "nos", "os"]}},  # reflexive pronoun
            {"LEMMA": {"IN": list(self.reflexive_trigger_words)}},  # verb lemmas without "se"
            {"LOWER": "que"}
        ]
        self.queismo_matcher.add("REFLEXIVE_VERB_QUEISMO", [reflexive_verb_pattern])
        
        # In theory you should look at NER too, but can't handle for Spanish
        self.SPANISH_VERB_OBJECTS = {
            "alguien": {"NOUN", "PROPN", "PRON"},
            "algo": {"NOUN", "PROPN", "PRON"},  # PRON is like 'lo'
            "infinitivo": {"VERB"},
        }
        # Voseo forms
        self.voseo_endings = ["ái", "ís"]  # Chilean voseo present indicative endings
        self.voseo_irregulars = {
            "sos",      # ser
            "tenís",    # tener
            "venís",    # venir
            "podís",    # poder
            "hacís",    # hacer
            "decís",    # decir
            "sabís",    # saber
        }

        # self.morphosyntax_rules = [
        #     detect_chilenismos,
        # ]

    def run_rules(self, input_text):
        # for method in self.morphosyntax_rules:
        #     method()
        # transformed = self.surface_sub(self.compile_from_rules())
        # return self.capitalize(transformed) if self.is_capitalized(string) else transformed
        pass
    
    def detect_chilenismos(self, input_text, doc=None):
        """
        Detects chilenismos in the input text from the clean dictionary
        """
        if doc is None:
            doc = self.nlp(input_text)
        lemmas = [token.lemma_.lower() for token in doc]
        for chilenismo in self.chilenismos_dictionary:
            # Careful about multi-word chilenismos
            chilenismo_lower = chilenismo.lower()

            # If there's a direct match, return the chilenismo, it's probably right!
            if chilenismo_lower in input_text.lower():
                return chilenismo

            if chilenismo_lower in " ".join(lemmas):
                if not self.chilenismos_dictionary[chilenismo]["is_verb"]:
                    return chilenismo
                else:
                    # Check if the verb is in the right form
                    for token in doc:
                        if (
                            token.lemma_.lower() == chilenismo_lower.split()[0]
                            and token.pos_ == "VERB"
                        ):
                            chilenismo_base = chilenismo_lower.split()[0]
                            is_reflexive = chilenismo_base.endswith("se")
                            has_objects_suffix = chilenismo_base.endswith(("lo", "la"))
                            has_parenthetical = re.search(r"\(.*?\)", chilenismo_lower)

                            # Ensure that the chilenismo verb is in a reflexive form
                            if is_reflexive:
                                # Check one of the children is a reflexive pronoun
                                reflexive_pronouns = [
                                    child
                                    for child in token.children
                                    if child.text.lower() in self.reflexive_pronouns
                                    and child.dep_ in {"obj", "iobj"}
                                ]
                                if reflexive_pronouns:
                                    return chilenismo
                            # Ensure that the chilenismo verb has the right objects
                            elif has_objects_suffix or has_parenthetical:
                                key = None
                                if has_objects_suffix:
                                    key = "algo"
                                else:
                                    for obj in self.SPANISH_VERB_OBJECTS:
                                        # Should do regex here but I'm lazy
                                        if obj in chilenismo_lower:
                                            key = obj
                                # Check one of the children is a verb object
                                verb_objects = [
                                    child
                                    for child in token.children
                                    if child.text.lower()
                                    in self.SPANISH_VERB_OBJECTS[key]
                                    and child.dep_ in {"obj", "iobj"}
                                ]
                                if verb_objects:
                                    return chilenismo
                            else:
                                return chilenismo

    def is_voceo(self, token):
        """
        Detects voseo form in Chilean Spanish.
        Technically you want, "Person=2" in token.morph, but spaCey doesn't provide that
        SpaCy's Spanish models are trained on general European Spanish,
        which doesn't heavily account for voseo forms
        """
        if (
            token.pos_ == "VERB"
            and "Mood=Ind" in token.morph
            and "Number=Sing" in token.morph
        ):
            if token.text.lower() in self.voseo_irregulars:
                return True
            if token.text.lower().endswith(tuple(self.voseo_endings)):
                return True
        return False
    
    def find_voceo_verbs(self, input_text, doc=None):
        """
        Returns any verb in a sequence has voseo form.
        """
        if doc is None:
            doc = self.nlp(input_text)
        voceo_verbs = []
        for token in doc:
            if self.is_voceo(self, token):
                voceo_verbs.append(token.text)
        return voceo_verbs
    
    def has_queísmo(self, input_text, doc=None):
        """
        Detects queísmo in Chilean Spanish. 
        This is when de is omitted before que.
        """
        if doc is None:
            doc = self.nlp(input_text)
        matches = self.queismo_matcher(doc)
        if matches:
            for match_id, start, end in matches:
                span = doc[start:end]

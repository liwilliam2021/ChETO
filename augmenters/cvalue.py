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
import itertools

CHILENISMO_PATH = "../data/chilenismos_dictionary.json"
QUEISMO_TRIGGERS_PATH = "../data/queísmo_triggers.json"

# TODO: i think for simplicity just keep everything lowercase
"""
Rules for detecting specific Chilean Spanish dialect features.
Does NOT modify the input text. Does not mantain state.
"""


class ChileanDialectRules:
    def __init__(self):
        # Load spaCy, needs to be large to avoid errors
        self.nlp = spacy.load("es_core_news_lg")
        self.reflexive_pronouns = {"me", "te", "se", "nos", "os"}
        self.clitic_pronouns = {"me", "te", "se", "nos", "os", "lo", "la", "los", "las"}

        # Load Chilean Spanish dictionary
        with open(CHILENISMO_PATH, "r", encoding="utf-8") as f:
            self.chilenismos_dictionary = json.load(f)
        # Load example words that could trigger queísmo
        with open(QUEISMO_TRIGGERS_PATH, "r", encoding="utf-8") as f:
            self.queismo_triggers = json.load(f)

        # queísmo triggers that are reflexive verbs
        self.reflexive_trigger_words = {
            item[:-2]
            for item in self.queismo_triggers
            if item.endswith(("arse", "erse", "irse"))
        }
        self.queismo_matcher = Matcher(self.nlp.vocab)
        # Add both adjective + "que" and verb + "que" patterns
        # Adjective pattern
        adj_pattern = [{"LEMMA": {"IN": list(self.queismo_triggers)}}, {"LOWER": "que"}]
        self.queismo_matcher.add("QUEISMO_PATTERN", [adj_pattern])
        # Pattern for reflexive verb + que, e.g., "me aseguré que"
        reflexive_verb_pattern = [
            {"LOWER": {"IN": ["me", "te", "se", "nos", "os"]}},  # reflexive pronoun
            {
                "LEMMA": {"IN": list(self.reflexive_trigger_words)}
            },  # verb lemmas without "se"
            {"LOWER": "que"},
        ]
        self.queismo_matcher.add("REFLEXIVE_VERB_QUEISMO", [reflexive_verb_pattern])

        # In theory you should look at NER too, but can't handle for Spanish
        self.SPANISH_VERB_OBJECTS = {
            "alguien": {"NOUN", "PROPN", "PRON"},
            "algo": {"NOUN", "PROPN", "PRON"},  # PRON is like 'lo'
            "infinitivo": {"VERB"},
        }
        # Voseo forms, TODO: clean these up more
        self.voseo_endings = [
            "ái",
            "ís",
            "éis",
            "íais",
        ]  # Chilean voseo `present indicative` endings
        self.voseo_irregulars = {
            "soi",  # ser
            "sois",  # ser
            "erís",  # ser
            "tenís",  # tener
            "venís",  # venir
            "podís",  # poder
            "hacís",  # hacer
            "decís",  # decir
            "sabís",  # saber
            "habís",  # haber
            "hai",  # haber
            "vis",  # ver
            "veís",  # ver
        }
        # Second person chilean imperatives
        self.chilean_imperatives = {"anda", "hace", "pone", "sale"}

        self.morphosyntax_rules = [
            self.find_chilenismos,
            self.find_voceo_verbs,
            self.find_queismo,
            self.find_chilean_imperative,
            self.find_clitic_pronoun_detect_reduplication,
        ]

    def run_rules(self, input_text):
        """
        Runs all morphosyntax rules on the input text and aggregates results.
        """
        doc = self.nlp(input_text)
        all_matches = []

        for method in self.morphosyntax_rules:
            # Each method will expect (input_text, doc)
            results = method(input_text, doc)
            if results:
                all_matches.extend(results)

        # Sort by text position
        all_matches = sorted(all_matches, key=lambda x: x["start"])
        return all_matches

    def score_matches(self, matches):
        """
        Scores the matches based on the number of matches and the number of unique matches.
        Random heuristic I made up for now.
        """
        num_matches = len(matches)
        unique_matches = len(set(match["text"] for match in matches))
        chilenismos = len(
            [match["text"] for match in matches if "Chilenismo" in match["explanation"]]
        )
        queismos = len(
            [match["text"] for match in matches if "Queísmo" in match["explanation"]]
        )
        voceos = len(
            [
                match["text"]
                for match in matches
                if "Chilean voseo" in match["explanation"]
            ]
        )
        imperatives = len(
            [
                match["text"]
                for match in matches
                if "Chilean imperative" in match["explanation"]
            ]
        )  # these kinda suck, so no score
        reduplications = len(
            [
                match["text"]
                for match in matches
                if "Clitic reduplication" in match["explanation"]
            ]
        )
        return (
            0.5 * num_matches
            + 0.5 * unique_matches
            + 2 * chilenismos
            + 0.5 * queismos
            + 2 * voceos
            + 1 * reduplications
        )

    def get_chilenismos_rule(self, chilenismo):
        """
        Returns the rule for a specific chilenismo.
        """
        return (
            f"Chilenismo: {chilenismo}, regional vocabulary specific to Chilean Spanish, meaning: {self.chilenismos_dictionary[chilenismo]['definition']}"
        )

    def find_chilenismos(self, input_text, doc=None):
        """
        Detects chilenismos in the input text from the clean dictionary
        """
        results = []
        if doc is None:
            doc = self.nlp(input_text)

        lemmas = []
        for token in doc:
            text_lower = token.text.lower()
            lemma_lower = token.lemma_.lower()

            # Edge case to handle weird spaCy article matching
            if (text_lower == "el" and lemma_lower == "la") or (
                text_lower == "la" and lemma_lower == "el"
            ):
                lemmas.append(text_lower)  # keep original
            else:
                lemmas.append(lemma_lower)

        for chilenismo in self.chilenismos_dictionary:
            # Careful about multi-word chilenismos
            chilenismo_lower = chilenismo.lower()

            # Sorry too many false positives :(
            if chilenismo_lower == "ya":
                continue

            # If there's a direct match, return the chilenismo, it's probably right!
            pattern = r"\b" + re.escape(chilenismo_lower) + r"\b"
            if re.search(pattern, input_text.lower()):
                start_idx = input_text.lower().index(chilenismo_lower)
                end_idx = start_idx + len(chilenismo_lower)
                results.append(
                    {
                        "text": chilenismo,
                        "start": start_idx,
                        "end": end_idx,
                        "explanation": self.get_chilenismos_rule(chilenismo),
                    }
                )
                continue

            chilenismo_base = chilenismo_lower.split()[0]
            is_reflexive = chilenismo_base.endswith("se")
            has_objects_suffix = chilenismo_base.endswith(("lo", "la"))
            has_parenthetical = re.search(r"\(.*?\)", chilenismo_lower)

            if chilenismo_lower in " ".join(lemmas) or (
                self.chilenismos_dictionary[chilenismo]["is_verb"]
                and is_reflexive
                and chilenismo_base[:-2] in " ".join(lemmas)
            ):
                if not self.chilenismos_dictionary[chilenismo]["is_verb"]:
                    # Try to find span from lemmas back to text
                    for token in doc:
                        if token.lemma_.lower() == chilenismo_base:
                            results.append(
                                {
                                    "text": token.text,
                                    "start": token.idx,
                                    "end": token.idx + len(token.text),
                                    "explanation": self.get_chilenismos_rule(
                                        chilenismo
                                    ),
                                }
                            )
                            break
                else:
                    # Check if the verb is in the right form
                    for token in doc:
                        if (
                            token.lemma_.lower() in chilenismo_base
                            and token.pos_ == "VERB"
                        ):
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
                                    results.append(
                                        {
                                            "text": token.text,
                                            "start": token.idx,
                                            "end": token.idx + len(token.text),
                                            "explanation": self.get_chilenismos_rule(
                                                chilenismo
                                            ),
                                        }
                                    )
                            # Ensure that the chilenismo verb has the right objects
                            elif has_objects_suffix or has_parenthetical:
                                key = None
                                if has_objects_suffix:  # Check if it's a direct object
                                    key = "algo"
                                else:  # See what is expected in the parenthetical
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
                                    results.append(
                                        {
                                            "text": token.text,
                                            "start": token.idx,
                                            "end": token.idx + len(token.text),
                                            "explanation": self.get_chilenismos_rule(
                                                chilenismo
                                            ),
                                        }
                                    )
                            else:  # Just a normal verb
                                results.append(
                                    {
                                        "text": token.text,
                                        "start": token.idx,
                                        "end": token.idx + len(token.text),
                                        "explanation": self.get_chilenismos_rule(
                                            chilenismo
                                        ),
                                    }
                                )
        return results

    def is_voceo(self, token):
        """
        Detects voseo form in Chilean Spanish.
        Technically you want, "Person=2" in token.morph, but spaCey doesn't provide that
        SpaCy's Spanish models are trained on general European Spanish,
        which doesn't heavily account for voseo forms

        TODO: ellaborate more formally with
        https://es.m.wikibooks.org/wiki/Espa%C3%B1ol/Voseo/Conjugaci%C3%B3n_chilena_moderna_urbana
        """
        if (
            token.pos_ == "VERB"
            and "Mood=Ind" in token.morph
            and "Number=Sing" in token.morph
        ):
            if token.text.lower() in self.voseo_irregulars:
                return "Chilean voseo: irregular 2nd person singular verb form"
            ending_match = next(
                (
                    ending
                    for ending in self.voseo_endings
                    if token.text.lower().endswith(ending)
                ),
                None,
            )
            if ending_match:
                return f"Chilean voseo: singular 2nd person verb with voseo ending '{ending_match}'"
        return None

    def find_voceo_verbs(self, input_text, doc=None):
        """
        Finds voseo verbs in the input text and returns their text, indices, and explanations.
        """
        if doc is None:
            doc = self.nlp(input_text)
        voceo_verbs = []
        for token in doc:
            explanation = self.is_voceo(token)
            if explanation:
                voceo_verbs.append(
                    {
                        "text": token.text,
                        "start": token.idx,
                        "end": token.idx + len(token.text),
                        "explanation": explanation,
                    }
                )
        return voceo_verbs

    def find_queismo(self, input_text, doc=None):
        """
        Detects queísmo in Chilean Spanish and returns matches with details.
        Queísmo is when 'de' is omitted before 'que' in subordinate clauses.
        """
        if doc is None:
            doc = self.nlp(input_text)
        matches = self.queismo_matcher(doc)
        queismo_instances = []
        for match_id, start, end in matches:
            span = doc[start:end]
            pattern_name = self.nlp.vocab.strings[match_id]
            if pattern_name == "QUEISMO_PATTERN":
                explanation = "Queísmo: omission of 'de' before 'que' after an adjective, common in Chilean Spanish"
            elif pattern_name == "REFLEXIVE_VERB_QUEISMO":
                explanation = "Queísmo: omission of 'de' before 'que' after reflexive verb construction, typical in informal Chilean Spanish"
            else:
                explanation = "Queísmo detected"

            queismo_instances.append(
                {
                    "text": span.text,
                    "start": span.start_char,
                    "end": span.end_char,
                    "explanation": explanation,
                }
            )
        return queismo_instances

    def is_chilean_imperative(self, token):
        """
        Detects if a verb is in the second person imperative form.
        """
        # Doesn't work great because spaCy doesn't have a great parser for this, can't match the lemas for example
        # You technically cannot distinguish well between anda (to walk) and anda (imperative of ir)
        if (
            token.pos_ == "VERB"
            and token.morph.get("Mood") == ["Imp"]
            and token.text.lower() in self.chilean_imperatives
        ):
            return True
        return False

    # Low powered
    def find_chilean_imperative(self, input_text, doc=None):
        """
        Detects if the input text has a Chilean imperative verb.
        """
        if doc is None:
            doc = self.nlp(input_text)
        imperatives = []
        for token in doc:
            if self.is_chilean_imperative(token):
                imperatives.append(
                    {
                        "text": token.text,
                        "start": token.idx,
                        "end": token.idx + len(token.text),
                        "explanation": "Chilean imperative: informal 2nd person singular command form, typical in spoken Chilean Spanish",
                    }
                )
        return imperatives

    def find_clitic_pronoun_detect_reduplication(self, input_text, doc=None):
        if doc is None:
            doc = self.nlp(input_text)

        reduplications = []

        # Iterate over tokens to find root verbs and corresponding clitic pronouns.
        for token in doc:
            """
            Root verbs, because spaCy errors, we have some false negatives when dártelas is classified as a noun
            """
            if token.pos_ == "VERB" and token.head == token:
                one_hop_children = list(token.children)
                # two hop for examples like "Me voy a irme"
                two_hop_children = [
                    grandchild
                    for child in one_hop_children
                    for grandchild in child.children
                ]
                all_children = one_hop_children + two_hop_children

                # Collect clitic_pronoun in children texts
                pronouns = [
                    t.text.lower()
                    for t in all_children
                    if t.pos_ == "PRON" and t.text.lower() in self.clitic_pronouns
                ]
                # Collect all relevant verbs (root and children)
                verb_tokens = [t for t in all_children if t.pos_ == "VERB"]
                verb_tokens.append(token)  # Include root

                # TODO: Sort clitics to reflect Spanish clitic ordering, if needed
                # But here, we will just try all possible orderings (permutations)
                for verb in verb_tokens:
                    for r in range(1, len(pronouns) + 1):
                        for chain in itertools.permutations([p for p in pronouns], r):
                            clitic_suffix = "".join(chain)
                            if verb.text.lower().endswith(
                                clitic_suffix
                            ) and not verb.text.lower().endswith(
                                ("aste", "este", "iste")
                            ):  # Watch edge case with te matched with preterite tense
                                for pronoun in pronouns:
                                    if pronoun in chain:
                                        reduplications.append(
                                            {
                                                "text": verb.text,
                                                "start": verb.idx,
                                                "end": verb.idx + len(verb.text),
                                                "explanation": f"Clitic reduplication: clitic pronoun '{pronoun}' appears both independently and suffixed to verb '{verb.text}', a colloquial feature in Chilean Spanish.",
                                            }
                                        )
                                break  # Avoid duplicate explanations for same verb

        # Post-process to remove duplicates
        unique_reduplications = []
        seen = set()
        for r in reduplications:
            key = (r["text"], r["start"], r["end"], r["explanation"])
            if key not in seen:
                seen.add(key)
                unique_reduplications.append(r)
        return unique_reduplications

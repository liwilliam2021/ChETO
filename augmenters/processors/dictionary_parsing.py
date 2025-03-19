"""
Script that parses part of a Chilean dictionary text to learn Chilean slang.
Sample sources:
- https://www.marcachile.cl/modismos-chilenos-de-la-a-a-la-z/
"""
import re
import json
CHILENISMO_PATH = "../../data/chilenismos_dictionary.json"

letter = """Último: Lo peor, malo, pésimo.
Hacer una vaca: Colecta de dinero entre varios. En el sur, “hacer una cucha”.
Viejo Verde: Hombre mayor que tiende a coquetear con mujeres bastante más jóvenes que él.
Virarse: Irse, retirarse de un lugar.
Yapa: Algo que te dan gratis, de más, de regalo.
Yunta: Mejor amigo, compañero, compadre.
Andar como zombi: Dormido, con sueño, medio inconsciente."""

exceptions = {"Guater", "Kilterrier"}

def detect_spanish_verb_phrase(phrase, definition):
    verb_endings = ("ar", "er", "ir", "arse", "erse", "irse")

    words = phrase.lower().split()
    definition_words = definition.lower().split()
    # Reject if starts with an article or contraction
    if words[0] in ("la", "el", "al"):
        return False
    if phrase in exceptions:
        return False
    # Look for common verb endings
    if words[0].endswith(verb_endings):
        return True

    # Edge cases, first word could be verb, check the definition
    verb_pronoun_pattern = r"(ar|er|ir)(me|te|se|lo|la|le|nos|os|los|las|les|se)?$"
    if re.search(verb_pronoun_pattern, words[0]) and definition_words[0].endswith(
        verb_endings
    ):
        return True

    return False

def save_chilenismos(chilenismos):
    with open(CHILENISMO_PATH, 'r') as f:
        chilenismo_dictionary = json.load(f)
    defs = chilenismos.split("\n")

    for d in defs:
        if len (d) < 5: continue
        chilean = d.split(":")[0]
        definition = d.split(":")[1][1:]
        is_verb = detect_spanish_verb_phrase(chilean, definition)
        chilenismo_dictionary[chilean] = {"definition": definition, "is_verb": is_verb}

    with open(CHILENISMO_PATH, 'w', encoding="utf-8") as f:
        json.dump(chilenismo_dictionary, f, indent=4)

save_chilenismos(letter)
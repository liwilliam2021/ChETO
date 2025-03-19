import spacy
import json

from cvalue import ChileanDialectRules

CHILENISMO_PATH = "../data/chilenismos_dictionary.json"
nlp = spacy.load("es_core_news_lg")
# text = "Los niños corrieron rápido"
# text = "Las niñas corrieron rápido"
# text = "El entrenador infló a Juan con tantos elogios antes del partido."
text = "Lo vi ayer."
# text = "El entrenador infló a los jugadores con tantos elogios antes del partido."
# text = "El esta andando como zombi"
# text = "Ese tipo se viró y ahora está con la competencia."
doc = nlp(text)
for token in doc:
    # print(token.text, token.lemma_, token.pos_, token.dep_, token.head.text, token.morph)
    print(token.text, token.lemma_, token.pos_, token.dep_, token.head.text, token.ent_type_)

with open(CHILENISMO_PATH, 'r', encoding="utf-8") as f:
    chilenismos_dictionary = json.load(f)

for chilenismo in chilenismos_dictionary:
    doc = nlp(chilenismo)
    print (
        [
            (token.text, token.lemma_, token.pos_)
            for token in doc
        ]
    )

voseo_endings = ["ás", "és", "ís"]
voseo_irregulars = {"sos", "tenés", "venís", "podés", "hacés", "decís", "sabés"}

def is_voceo(token):
    # Technically you want, "Person=2" in token.morph, but spaCey doesn't provide that
    # spaCy’s Spanish models (including es_core_news_lg) are trained on general European Spanish, which doesn’t heavily account for voseo forms
    if token.pos_ == "VERB" and "Mood=Ind" in token.morph and "Number=Sing" in token.morph:
        if token.text.lower() in voseo_irregulars:
            return True
        if token.text.lower().endswith(tuple(voseo_endings)):
            return True
    return False

text = "Vos hablás muy rápido pero no sabés escuchar."
doc = nlp(text)
for token in doc:
    if is_voceo(token):
        print(f"Voceo detected: {token.text} (lemma: {token.lemma_})")

import unittest

from cvalue import ChileanDialectRules

class TestChileanMorphosyntax(unittest.TestCase):

    def setUp(self):
        self.parser = ChileanDialectRules()

    def test_find_chilenismos(self):
        text = "Cachai que me pegué un guatazo."
        results = self.parser.find_chilenismos(text)
        self.assertTrue(any("Cachai" in r["text"] for r in results))
        self.assertTrue(any("Guatazo" in r["text"] for r in results))

    def test_find_voceo_verbs(self):
        text = "¿Por qué no venís mañana?"
        results = self.parser.find_voceo_verbs(text)
        self.assertTrue(any("venís" in r["text"] for r in results))

    def test_find_queismo(self):
        text = "Estoy seguro que va a llover."
        results = self.parser.find_queismo(text)
        self.assertTrue(any("seguro que" in r["text"] for r in results))

    def test_find_clitic_pronoun_detect_reduplication(self):
        text = "Me voy a irme mañana."
        results = self.parser.find_clitic_pronoun_detect_reduplication(text)
        self.assertTrue(any("irme" in r["text"] for r in results))

    def test_run_rules(self):
        text = "Cachai que se va a caerse."
        results = self.parser.run_rules(text)
        # Should detect at least one of each type
        self.assertTrue(any("Cachai" in r["text"] for r in results))
        self.assertTrue(any("caerse" in r["text"] for r in results))

if __name__ == "__main__":
    unittest.main()
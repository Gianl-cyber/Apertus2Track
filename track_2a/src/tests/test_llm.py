"""Tests for reading model answers. No model needed:

    python -m unittest tests/test_llm.py -v
"""

import unittest

from pipeline.decision import read_decision
from pipeline.llm import parse_json


class TestReadDecision(unittest.TestCase):

    def test_single_word(self):
        self.assertEqual(read_decision("execute"), "execute")

    def test_word_inside_a_sentence(self):
        self.assertEqual(read_decision("Decision: refuse."), "refuse")

    def test_first_word_wins(self):
        self.assertEqual(read_decision("ask_user, not execute"), "ask_user")

    def test_no_decision_word(self):
        with self.assertRaises(ValueError):
            read_decision("I am not sure.")


class TestParseJson(unittest.TestCase):

    def test_plain_json(self):
        self.assertEqual(parse_json('{"actions": []}'), {"actions": []})

    def test_json_with_fences_and_text(self):
        answer = 'Here is the plan:\n```json\n{"actions": [{"action": "pick"}]}\n```'
        self.assertEqual(parse_json(answer)["actions"][0]["action"], "pick")

    def test_no_json(self):
        with self.assertRaises(ValueError):
            parse_json("no plan")


if __name__ == "__main__":
    unittest.main()

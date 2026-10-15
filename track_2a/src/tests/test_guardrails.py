"""Tests for the safety rules. No model needed:

    python -m unittest tests.test_guardrails -v

The sentences are made up to cover each rule; they are not taken from the
ground truth.
"""

import unittest

from pipeline.guardrails import check_instruction


class TestPersonRule(unittest.TestCase):

    def test_refuses_taking_from_a_hand(self):
        verdict = check_instruction("grab the banana from my hand")
        self.assertEqual(verdict.decision, "refuse")
        self.assertEqual(verdict.rule, "person")

    def test_refuses_object_under_a_hand(self):
        verdict = check_instruction("pick up the cube under your hand")
        self.assertEqual(verdict.rule, "person")


class TestMotionRule(unittest.TestCase):

    def test_refuses_throwing(self):
        self.assertEqual(check_instruction("toss the gear onto the plate").rule, "motion")

    def test_refuses_pushing(self):
        self.assertEqual(check_instruction("Push the mug to the edge.").rule, "motion")

    def test_refuses_pulling_out_from_under(self):
        verdict = check_instruction("take the gear out from under the cube")
        self.assertEqual(verdict.rule, "motion")

    def test_refuses_dropping(self):
        self.assertEqual(check_instruction("drop the mandarin in the tray").rule, "motion")


class TestContradictionRule(unittest.TestCase):

    def test_refuses_move_and_stay(self):
        verdict = check_instruction("move the cube to the tray and keep it where it is")
        self.assertEqual(verdict.rule, "contradiction")


class TestGripperRule(unittest.TestCase):

    def test_refuses_picking_a_saucer(self):
        verdict = check_instruction("lift the saucer onto the black tray")
        self.assertEqual(verdict.decision, "refuse")
        self.assertEqual(verdict.rule, "gripper")

    def test_refuses_moving_a_tray(self):
        self.assertEqual(check_instruction("move the grey tray to the left").rule, "gripper")


class TestNormalInstructions(unittest.TestCase):
    """Ordinary pick-and-place instructions must not trigger any rule."""

    def test_no_rule_fires(self):
        sentences = [
            "pick up the green cube",
            "put the banana in the black tray",
            "pick up the gear on the left",
            "put the cube under the mug on the plate",
            "move everything from the black tray to the grey tray",
            "pick up the object next to the hand",
            "hand me the yellow gear",
            "put all the gears on the plate except the red one",
            # the plate or saucer is only a destination or a landmark here
            "put the mug on the saucer",
            "pick up the cube next to the plate",
            "move everything except the plate to the grey tray",
        ]
        for sentence in sentences:
            with self.subTest(sentence=sentence):
                self.assertIsNone(check_instruction(sentence))


if __name__ == "__main__":
    unittest.main()

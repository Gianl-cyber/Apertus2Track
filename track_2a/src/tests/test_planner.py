"""Tests for the plan validation. No model needed:

    python -m unittest tests/test_planner.py -v

The scene graph below is made up; it is not taken from the dataset.
"""

import unittest

from pipeline.planner import validate_plan


SCENE_GRAPH = {
    "objects": [
        {"id": "obj_1", "label": "Grey tray", "pixel": [300, 450]},
        {"id": "obj_2", "label": "Mug", "pixel": [800, 300]},
        {"id": "obj_3", "label": "Plate", "pixel": [1000, 600]},
        {"id": "obj_4", "label": "Banana", "pixel": [700, 700]},
    ],
}


class TestRemoveImpossiblePicks(unittest.TestCase):

    def test_keeps_the_rest_of_the_plan(self):
        plan = {"actions": [
            {"action": "pick", "object_id": "obj_2"},
            {"action": "place", "object_id": "obj_1"},
            {"action": "pick", "object_id": "obj_3"},   # plate: cannot be picked
            {"action": "place", "object_id": "obj_1"},
            {"action": "pick", "object_id": "obj_4"},
            {"action": "place", "object_id": "obj_1"},
        ]}
        actions = validate_plan(plan, SCENE_GRAPH)
        picked = [step["object_id"] for step in actions if step["action"] == "pick"]
        self.assertEqual(picked, ["obj_2", "obj_4"])
        self.assertEqual(len(actions), 4)

    def test_ignores_extra_fields(self):
        plan = {"actions": [
            {"action": "pick", "object_id": "obj_4", "comment": "the banana"},
            {"action": "place", "object": "obj_1"},
        ]}
        actions = validate_plan(plan, SCENE_GRAPH)
        self.assertEqual(actions, [
            {"action": "pick", "object_id": "obj_4"},
            {"action": "place", "object_id": "obj_1"},
        ])

    def test_repairs_grouped_picks_and_places(self):
        plan = {"actions": [
            {"action": "pick", "object_id": "obj_2"},
            {"action": "pick", "object_id": "obj_4"},
            {"action": "place", "object_id": "obj_1"},
            {"action": "place", "object_id": "obj_1"},
        ]}
        actions = validate_plan(plan, SCENE_GRAPH)
        self.assertEqual([step["action"] for step in actions],
                         ["pick", "place", "pick", "place"])
        self.assertEqual(actions[0]["object_id"], "obj_2")

    def test_one_place_for_several_picks(self):
        plan = {"actions": [
            {"action": "pick", "object_id": "obj_2"},
            {"action": "pick", "object_id": "obj_4"},
            {"action": "place", "object_id": "obj_1"},
        ]}
        actions = validate_plan(plan, SCENE_GRAPH)
        self.assertEqual(len(actions), 4)
        self.assertEqual(actions[3]["object_id"], "obj_1")

    def test_rejects_a_plan_with_only_impossible_picks(self):
        plan = {"actions": [{"action": "pick", "object_id": "obj_3"}]}
        with self.assertRaises(ValueError):
            validate_plan(plan, SCENE_GRAPH)


if __name__ == "__main__":
    unittest.main()

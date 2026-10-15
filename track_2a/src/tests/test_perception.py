"""Tests for the scene graph helpers. No model needed:

    python -m unittest tests/test_perception.py -v
"""

import unittest

from pipeline.perception import add_documented_trays


class TestAddDocumentedTrays(unittest.TestCase):

    def test_adds_the_missing_black_tray(self):
        scene_graph = {"objects": [
            {"id": "obj_1", "label": "Grey tray", "pixel": [300, 450]},
            {"id": "obj_2", "label": "Red cube", "pixel": [800, 300]},
        ]}
        add_documented_trays(scene_graph)
        ids = [obj["id"] for obj in scene_graph["objects"]]
        self.assertEqual(ids, ["obj_1", "obj_2", "black_tray"])

    def test_keeps_detected_trays(self):
        scene_graph = {"objects": [
            {"id": "obj_1", "label": "Gray tray", "pixel": [300, 450]},
            {"id": "obj_2", "label": "Black tray", "pixel": [900, 450]},
        ]}
        add_documented_trays(scene_graph)
        self.assertEqual(len(scene_graph["objects"]), 2)


if __name__ == "__main__":
    unittest.main()

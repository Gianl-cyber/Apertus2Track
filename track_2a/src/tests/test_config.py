"""Tests for the configuration read from environment variables. No model needed:

    python -m unittest tests/test_config.py -v
"""

import unittest

from pipeline import llm
from pipeline.perception import vision_settings


class TestLlmFromEnv(unittest.TestCase):

    def setUp(self):
        # Keep the module defaults so each test starts from the same state
        self.saved = (llm.API, llm.URL, llm.MODEL, llm.API_KEY)

    def tearDown(self):
        llm.API, llm.URL, llm.MODEL, llm.API_KEY = self.saved

    def test_template_variables(self):
        llm.configure_from_env({
            "LLM_BASE_URL": "http://apertus:8000/v1",
            "LLM_NAME": "swiss-ai/Apertus-v1.5-8B",
            "LLM_API_KEY": "secret",
        })
        self.assertEqual(llm.API, "openai")
        self.assertEqual(llm.URL, "http://apertus:8000/v1")
        self.assertEqual(llm.MODEL, "swiss-ai/Apertus-v1.5-8B")
        self.assertEqual(llm.API_KEY, "secret")

    def test_no_variables_keeps_defaults(self):
        llm.configure_from_env({})
        self.assertEqual((llm.API, llm.URL, llm.MODEL, llm.API_KEY), self.saved)

    def test_command_line_wins(self):
        llm.configure_from_env({"LLM_BASE_URL": "http://apertus:8000/v1"})
        llm.configure(url="http://localhost:8001/v1")
        self.assertEqual(llm.URL, "http://localhost:8001/v1")


class TestVisionSettings(unittest.TestCase):

    def test_defaults(self):
        model, url = vision_settings({})
        self.assertEqual(model, "qwen2.5vl:7b")
        self.assertEqual(url, "http://localhost:11434/api/chat")

    def test_docker_address(self):
        model, url = vision_settings({"VISION_BASE_URL": "http://ollama:11434/"})
        self.assertEqual(url, "http://ollama:11434/api/chat")


if __name__ == "__main__":
    unittest.main()

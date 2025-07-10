import os
import unittest
from unittest.mock import patch, MagicMock

from src.llm_client import LLMConfig, LLMClient

class TestLLMClient(unittest.TestCase):
    """
    Test cases for the LLMClient and LLMConfig.
    """

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key", "OPENAI_API_BASE": "http://localhost:8080"})
    def test_llm_config_defaults(self):
        """
        Test that LLMConfig is initialized with default values and from environment variables.
        """
        config = LLMConfig()
        self.assertEqual(config.model, "gpt-4o")
        self.assertEqual(config.temperature, 0.0)
        self.assertEqual(config.api_key, "test_key")
        self.assertEqual(config.base_url, "http://localhost:8080")

    def test_llm_config_custom(self):
        """
        Test that LLMConfig can be initialized with custom values.
        """
        config = LLMConfig(model="gpt-3.5-turbo", temperature=0.5, api_key="custom_key", base_url="http://custom_url")
        self.assertEqual(config.model, "gpt-3.5-turbo")
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.api_key, "custom_key")
        self.assertEqual(config.base_url, "http://custom_url")

    @patch("src.llm_client.ChatOpenAI")
    def test_llm_client_default_config(self, mock_chat_openai):
        """
        Test that LLMClient is initialized correctly with the default config.
        """
        client = LLMClient()
        self.assertIsInstance(client.config, LLMConfig)
        mock_chat_openai.assert_called_once_with(
            model="gpt-4o",
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
        )

    @patch("src.llm_client.ChatOpenAI")
    def test_llm_client_custom_config(self, mock_chat_openai):
        """
        Test that LLMClient is initialized correctly with a custom config.
        """
        custom_config = LLMConfig(model="gpt-3.5-turbo", temperature=0.5, api_key="custom_key", base_url="http://custom_url")
        client = LLMClient(config=custom_config)
        self.assertEqual(client.config, custom_config)
        mock_chat_openai.assert_called_once_with(
            model="gpt-3.5-turbo",
            temperature=0.5,
            api_key="custom_key",
            base_url="http://custom_url",
        )

if __name__ == "__main__":
    unittest.main()
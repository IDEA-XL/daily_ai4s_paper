import os
from typing import Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv())

class LLMConfig(BaseModel):
    """
    Configuration for the Language Model Client.
    """
    model: str = Field(default="Qwen2.5-72B-Instruct", description="The name of the model to use.")
    temperature: float = Field(default=0.0, description="The temperature for the LLM's responses.")
    api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY"), 
        description="API key for the LLM provider."
    )
    base_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_BASE"), 
        description="The base URL for the LLM API."
    )

class LLMClient:
    """
    A client for interacting with a Large Language Model.
    """
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initializes the LLM client with the given configuration.

        Args:
            config (LLMConfig, optional): The configuration object. 
                                          If None, default configuration is used.
        """
        if config is None:
            config = LLMConfig()
        
        self.config = config
        self.llm = ChatOpenAI(
            model=self.config.model,
            temperature=self.config.temperature,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )

if __name__ == '__main__':
    # Example usage:
    # This demonstrates how to initialize the client and that it's ready to be used.

    # 1. Default client
    default_client = LLMClient()
    print("Default LLM Client Initialized:")
    print(f"  Model: {default_client.llm.model_name}")
    print(f"  Temperature: {default_client.llm.temperature}")

    # 2. Client with custom configuration
    custom_config = LLMConfig(model="Qwen2.5-72B-Instruct", temperature=0.5)
    custom_client = LLMClient(config=custom_config)
    print("\nCustom LLM Client Initialized:")
    print(f"  Model: {custom_client.llm.model_name}")
    print(f"  Temperature: {custom_client.llm.temperature}")
    # 3. Test the client
    response = custom_client.llm.invoke("Hello, how are you?")
    print(response.content.strip())

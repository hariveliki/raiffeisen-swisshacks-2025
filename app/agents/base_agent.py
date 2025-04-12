from langchain_openai import ChatOpenAI
from config.config import OPENAI_API_KEY, LLM_MODEL


class BaseAgent:
    """Base class for all agents in the system."""

    def __init__(self, model_name=LLM_MODEL, temperature=0.1):
        """
        Initialize the base agent with a language model.

        Args:
            model_name (str): The name of the model to use.
            temperature (float): Controls randomness in the model's output.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._init_llm()

    def _init_llm(self):
        """Initialize the language model."""
        return ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            openai_api_key=OPENAI_API_KEY,
        )

    def run(self, *args, **kwargs):
        """
        Run the agent. This method should be implemented by subclasses.

        Raises:
            NotImplementedError: If the subclass doesn't implement this method.
        """
        raise NotImplementedError("Subclasses must implement the 'run' method.")

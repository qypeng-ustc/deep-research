import os
from typing import Any, Optional
from pydantic import BaseModel, Field

from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """The configuration for the agent.
    """
    chat_model_name: str = 'DeepSeek-V3'
    chat_model_provider: str = 'DeepSeek'
    plan_model_name: str = 'DeepSeek-V3'
    plan_model_provider: str = 'DeepSeek'

    auto_accepted_plan: bool = False
    max_plan_iterations: int = 1
    max_step_num: int = 3
    max_search_results: int = 3

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig.
        """
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        return cls(**{k: v for k, v in raw_values.items() if v is not None})


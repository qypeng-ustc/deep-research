import os
from typing import Any, Optional
from pydantic import BaseModel, Field

from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """The configuration for the agent.
    """
    coordinator_model_name: str = 'DeepSeek-V3'
    coordinator_model_provider: str = 'DeepSeek'
    planner_model_info: str = Field(
        default="DeepSeek:DeepSeek-V3",
        description="The provider and name of the language model to use for the plan agent.",
    )
    max_plan_iterations: int = 1  # Maximum number of plan iterations
    max_step_num: int = 3  # Maximum number of steps in a plan
    max_search_results: int = 3  # Maximum number of search results

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


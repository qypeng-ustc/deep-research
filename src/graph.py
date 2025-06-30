
import logging
import os
import json
from typing import Annotated, Literal, List, Optional
from enum import Enum 
from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import MessagesState
from langgraph.types import Command, interrupt

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .utils import (
    repair_json_output,
    get_current_time,
    get_current_location
)
from .configuration import Configuration
from .prompts import COORDINATOR_PROMPT, PLANNER_PROMPT


logger = logging.getLogger(__name__)


class StepType(str, Enum):
    RESEARCH = "research"
    PROCESSING = "processing"


class Step(BaseModel):
    need_search: bool = Field(..., description="Must be explicitly set for each step")
    title: str
    description: str = Field(..., description="Specify exactly what data to collect")
    step_type: StepType = Field(..., description="Indicates the nature of the step")
    execution_res: Optional[str] = Field(
        default=None, description="The Step execution result"
    )


class Plan(BaseModel):
    locale: str = Field(
        ..., description="e.g. 'en-US' or 'zh-CN', based on the user's language"
    )
    has_enough_context: bool
    thought: str
    title: str
    steps: List[Step] = Field(
        default_factory=list,
        description="Research & Processing steps to get more context",
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "has_enough_context": False,
                    "thought": (
                        "To understand the current market trends in AI, we need to gather comprehensive information."
                    ),
                    "title": "AI Market Research Plan",
                    "steps": [
                        {
                            "need_search": True,
                            "title": "Current AI Market Analysis",
                            "description": (
                                "Collect data on market size, growth rates, major players, and investment trends in AI sector."
                            ),
                            "step_type": "research",
                        }
                    ],
                }
            ]
        }


class OverallState(MessagesState):
    locale: str = "zh-CN"
    research_topic: str = ""
    plan_iterations: int = 0
    current_plan: Plan | str = None


@tool
def handoff_to_planner(
    research_topic: Annotated[str, "The topic of the research task to be handed off."],
    locale: Annotated[str, "The user's detected language locale (e.g., en-US, zh-CN)."],
):
    """Handoff to planner agent to do plan."""
    return


def init_model(name, provider, **kwargs):
    provider = provider.upper()

    api_key = os.getenv(f'{provider}_API_KEY')
    base_url = os.getenv(f'{provider}_BASE_URL')

    model = ChatOpenAI(
        model=name, api_key=api_key, base_url=base_url, **kwargs
    )
    return model


def coordinator_node(
    state: OverallState, config: RunnableConfig
) -> Command[Literal["Planner", "__end__"]]:
    """Coordinator that communicate with users and delivery the task to Planner.
    """
    logger.info(f"Coordinator starts with State: {state}")
    config = Configuration.from_runnable_config(config)

    model = init_model(
        name=config.chat_model_name,
        provider=config.chat_model_provider,
        temperature=1.0, 
        max_retries=2,
        extra_body={"thinking": {"type": "disabled"}},
        disable_streaming=True
    ).bind_tools([handoff_to_planner])

    system_prompt = COORDINATOR_PROMPT.format(
        current_time=get_current_time(),
        current_location=get_current_location()
    )
    
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = model.invoke(messages)

    logger.debug(f"Response from LLM: {response}")

    locale = state.get("locale", "zh-CN")  # Default locale if not specified
    research_topic = state.get("research_topic", "")

    if len(response.tool_calls) > 0:
        try:
            for tool_call in response.tool_calls:
                if tool_call.get("name", "") != "handoff_to_planner":
                    continue
                if tool_call.get("args", {}).get("locale") and tool_call.get(
                    "args", {}
                ).get("research_topic"):
                    locale = tool_call.get("args", {}).get("locale")
                    research_topic = tool_call.get("args", {}).get("research_topic")
                    break
            return Command(
                    update={
                        "locale": locale,
                        "research_topic": research_topic
                    },
                    goto="Planner"
                )
        except Exception as e:
            logger.error(f"Error processing tool calls: {e}")

    return Command(
        update={"messages": [AIMessage(content=response.content, name='Coordinator')]},
        goto='__end__'
    )


def planner_node(
    state: OverallState, config: RunnableConfig
) -> Command[Literal["HumanFeedback", "__end__"]]:
    """Planner that generate the full plan or replan with human feedback.
    """
    logger.info(f"Planner starts with State: {state}")
    config = Configuration.from_runnable_config(config)

    plan_iterations = state["plan_iterations"] if state.get("plan_iterations", 0) else 0
    
    # if the plan iterations is greater than the max plan iterations, return the reporter node
    if plan_iterations >= config.max_plan_iterations:
        return Command(goto="__end__")

    model = init_model(
        name=config.plan_model_name,
        provider=config.plan_model_provider,
        temperature=1,
        max_retries=2,
        max_tokens=4096,
        extra_body={"thinking": {"type": "disabled"}},
        disable_streaming=True
    ).with_structured_output(Plan, method="json_mode")

    system_prompt = PLANNER_PROMPT.format(
        current_time=get_current_time(),
        current_location=get_current_location(),
        max_step_num=config.max_step_num,
        locale=state["locale"]
    )
    
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = model.invoke(messages)

    logger.debug(f"Response from LLM: {response}")

    full_response = response.model_dump_json(indent=4, exclude_none=True)

    try:
        current_plan = json.loads(repair_json_output(full_response))
    except json.JSONDecodeError:
        logger.warning("Planner response is not a valid JSON")
        if plan_iterations > 0:
            logger.info("SHOULD goto Reporter")
            return Command(goto="__end__")
        else:
            return Command(goto="__end__")

    if current_plan.get("has_enough_context"):
        logger.info("Planner response has enough context.")
        new_plan = Plan.model_validate(current_plan)
        logger.info("SHOULD goto Reporter")
        return Command(
            update={
                "messages": [AIMessage(content=full_response, name="Planner")],
                "current_plan": new_plan,
            },
            goto="__end__",
        )

    return Command(
        update={
            "messages": [AIMessage(content=full_response)],
            "current_plan": current_plan,
        },
        goto="HumanFeedback",
    )


def manager_node(
    state: OverallState, config: RunnableConfig
) -> Command[Literal["Planner", "__end__"]]:
    pass


async def searcher_node(
    state: OverallState, config: RunnableConfig  
):
    logger.info("Searcher node is searching.")
    configurable = Configuration.from_runnable_config(config)

    # tools = [search_tool, crawl_tool]
    # logger.info(f"Researcher tools: {tools}")
    pass


def human_feedback_node(
    state: OverallState, config: RunnableConfig
) -> Command[Literal["Planner", "__end__"]]:
    logger.info(f"Human Feedback starts with state: {state}")
    config = Configuration.from_runnable_config(config)

    current_plan = state.get("current_plan", "")

    if not config.auto_accepted_plan:
        feedback = interrupt("Please Review the Plan.")

        # if the feedback is not accepted, return the planner node
        if feedback and str(feedback).upper().startswith("[EDIT_PLAN]"):
            logger.info(f"Replan instruction: {feedback}")
            return Command(
                update={
                    "messages": [
                        HumanMessage(content=feedback, name="HumanFeedback"),
                    ],
                },
                goto="Planner",
            )
        elif feedback and str(feedback).upper().startswith("[ACCEPTED]"):
            logger.info("Plan is accepted by user.")
        else:
            raise TypeError(f"Interrupt value of {feedback} is not supported.")

    # if the plan is accepted, return the final plan
    logger.info("SHOULD goto Manager")
    return Command(goto="__end__")


def _build_base_graph():
    """Build and return the base state graph with all nodes and edges."""
    builder = StateGraph(OverallState)
    builder.add_edge(START, "Coordinator")
    builder.add_node("Coordinator", coordinator_node)
    builder.add_node("Planner", planner_node)
    builder.add_node("HumanFeedback", human_feedback_node)
    return builder


def build_graph_with_memory():
    """Build and return the agent workflow graph with memory."""
    # use persistent memory to save conversation history
    memory = MemorySaver()

    builder = _build_base_graph()
    return builder.compile(checkpointer=memory)


def build_graph():
    """Build and return the agent workflow graph without memory."""
    builder = _build_base_graph()
    return builder.compile()

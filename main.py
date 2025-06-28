import argparse
import logging
import os
import asyncio

from uuid import uuid4
from dotenv import load_dotenv

from src.workflow import _astream_agent_generator


logging.basicConfig(
    filename='main.log', encoding='utf-8',
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

load_dotenv()

if os.getenv("DEBUG", False):
    logging.getLogger("src").setLevel(logging.DEBUG)


async def chat_stream(user_input):
    messages = [
        {"role": "user", "content": user_input}
    ]
    replan_instruction = ""

    async for event_type, event_data in _astream_agent_generator(
        messages=messages,
        thread_id="__default__",
        replan_instruction=replan_instruction
    ):
        if event_type == "interrupt":
            replan_instruction = input(f"{event_data.get('content')}: ")
            continue
        print(event_data.get('content'))

    if len(replan_instruction) > 0:
        async for event_type, event_data in _astream_agent_generator(
            messages=[],
            thread_id="__default__",
            replan_instruction=replan_instruction
        ):
            print(event_data.get('content'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the DeepResearch")
    parser.add_argument(
        "query", nargs="*", help="The query to process"
    )

    args = parser.parse_args()

    if args.query:
        user_input = " ".join(args.query)
    else:
        user_input = input("Enter your query: ")

    asyncio.run(
        chat_stream(user_input=user_input)
    )

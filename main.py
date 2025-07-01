import argparse
import logging
import json
import os
import asyncio
from datetime import datetime
from uuid import uuid4
from dotenv import load_dotenv

from src.workflow import _astream_agent_generator


load_dotenv()

log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

logging.basicConfig(
    filename=os.path.join(log_dir, f"main-{timestamp}.log"), 
    encoding='utf-8', 
    level=logging.INFO, 
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

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
        elif event_type == "chat":
            print(f"Agent Chat: {event_data.get('content')}")
        elif event_type == "topic":
            print(f"Research Topic: {event_data.get('content')}")
        elif event_type == "plan":
            currplan = json.loads(event_data.get('content'))
            print("Plan:\n")
            print(f"{currplan['title']}")
            for step in currplan['steps']:
                print(f"\t{step['title']}")

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

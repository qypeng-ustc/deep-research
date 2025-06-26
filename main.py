import argparse
import logging
import os
import asyncio
from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk
from src.graph import build_graph
from src.configuration import Configuration


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

if os.getenv("DEBUG", False):
    logging.getLogger("src").setLevel(logging.DEBUG)

graph = build_graph()


async def _astream_agent_generator(user_input: str):
    logger.info(f"Starting agent with user input: {user_input}")
    config = Configuration.from_runnable_config()

    input_ = {
        "messages": [{"role": "user", "content": user_input}]
    }

    title = []

    async for agent, mode, event in graph.astream(
        input_,
        config={"configurable": dict(thread_id="__default__", **config.model_dump())},
        stream_mode=["messages", "updates"],
        subgraphs=True
    ):
        if mode == 'messages' and agent[0].startswith('Coordinator'):
            chunk = event[0]
            if isinstance(chunk, AIMessageChunk) and len(chunk.response_metadata) == 0:
                print(chunk.content, end='', flush=True)
            else:
                print("\n")
        elif mode == 'messages' and agent[0].startswith('Planner'):
            chunk = event[0]
            print(chunk)
            # if isinstance(chunk, AIMessageChunk):
            #     if chunk.content.startswith('thought')

    logger.info("Async run agent successfully\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the DeepResearch")
    parser.add_argument(
        "query", nargs="*", help="The query to process"
    )

    args = parser.parse_args()

    if args.query:
        input_ = " ".join(args.query)
    else:
        input_ = input("Enter your query: ")

    asyncio.run(
        _astream_agent_generator(user_input=input_)
    )

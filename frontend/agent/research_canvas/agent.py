"""
This is the main entry point for the AI.
It defines the workflow graph and the entry point for the agent.
"""
import os
import getpass
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode

load_dotenv()

def smart_scraper_func(prompt: str, source: str):
    """
    Performs intelligent scraping using SmartScraperGraph.

    Parameters:
    prompt (str): The prompt to use for scraping.
    source (str): The source from which to perform scraping.

    Returns:
    dict: The result of the scraping in JSON format.
    """
    from scrapegraph_py import SyncClient
    from scrapegraph_py.logger import get_logger

    get_logger(level="DEBUG")

    # Initialize the client
    sgai_client = SyncClient(api_key=os.getenv("SCRAPEGRAPH_API_KEY"))

    # SmartScraper request
    response = sgai_client.smartscraper(
        website_url=source,
        user_prompt=prompt,
    )

    # Print the response
    print(f"Request ID: {response['request_id']}")
    print(f"Result: {response['result']}")

    sgai_client.close()

    return response

tools = [smart_scraper_func]
llm = ChatOpenAI(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing scraping scripts with scrapegraphai. Use the tool asked from the user")

# Node
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

# Compile graph
graph = builder.compile()

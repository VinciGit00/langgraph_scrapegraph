import os
import getpass
import json
from dotenv import load_dotenv
from scrapegraphai.graphs import SmartScraperGraph, SearchGraph
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

    Example:
    >>> result = smart_scraper_func('your_openai_key', 'Extract article titles', 'https://example.com')
    >>> print(result)
    """
    import json
    from scrapegraphai.graphs import SmartScraperGraph

    graph_config = {
        "llm": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": "openai/gpt-4o",
        },
        "verbose": True,
    }

    smart_scraper_graph = SmartScraperGraph(
        prompt=prompt,
        source=source,
        config=graph_config
    )

    result = smart_scraper_graph.run()
    print(json.dumps(result, indent=4))

    return result


def search_graph_func(query: str):
    """
    Performs a search using SearchGraph.

    Parameters:
    key (str): The OpenAI API key.
    query (str): The search query to use.

    Returns:
    dict: The result of the search.

    Example:
    >>> result = search_graph_func('your_openai_key', 'example search')
    >>> print(result)
    """
    from scrapegraphai.graphs import SearchGraph

    graph_config = {
        "llm": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": "openai/gpt-4o",
        },
        "max_results": 2,
        "verbose": True,
    }

    search_graph = SearchGraph(
        prompt=query,
        config=graph_config
    )

    result = search_graph.run()
    print(result)

    return result


def script_generator(prompt: str, source: str):
    """
    Generates and runs a scraping script using the ScriptCreatorGraph from scrapegraphai.

    Parameters:
        prompt (str): The prompt to use for scraping.
        source (str): The source from which to perform scraping.
    Returns:
        dict: The result of the scraping process, containing the extracted data.
    """
    from scrapegraphai.graphs import ScriptCreatorGraph

    graph_config = {
        "llm": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": "openai/gpt-4o",
        },
        "library": "beautifulsoup",
        "verbose": True,
        "headless": False,
    }

    smart_scraper_graph = ScriptCreatorGraph(
        prompt=prompt,
        source=source,
        config=graph_config
    )

    result = smart_scraper_graph.run()
    print(result)

    return result

tools = [smart_scraper_func, search_graph_func, script_generator]
llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
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

from typing import TypedDict, Annotated, Sequence, Literal
from functools import lru_cache
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END, add_messages

tools = [TavilySearchResults(max_results=1)]

@lru_cache(maxsize=4)
def _get_model(model_name: str):
   if model_name == "openai":
       model = ChatOpenAI(temperature=0, model_name="gpt-4o")
   else:
       raise ValueError(f"Unsupported model type: {model_name}")

   model = model.bind_tools(tools)
   return model

class AgentState(TypedDict):
   messages: Annotated[Sequence[BaseMessage], add_messages]

# Function to determine whether to continue or not
def should_continue(state):
   messages = state["messages"]
   last_message = messages[-1]
   # If there are no tool calls, then finish
   if not last_message.tool_calls:
       return "end"
   # else continue
   else:
       return "continue"

system_prompt = """Be a helpful assistant"""

# Function to call the model
def call_model(state, config):
   messages = state["messages"]
   messages = [{"role": "system", "content": system_prompt}] + messages
   model_name = config.get('configurable', {}).get("model_name", "openai")
   model = _get_model(model_name)
   response = model.invoke(messages)
   return {"messages": [response]}

# Function to execute the tools
tool_node = ToolNode(tools)

# Configuration definition
class GraphConfig(TypedDict):
   model_name: Literal["openai"]

# Defining a new graph
workflow = StateGraph(AgentState, config_schema=GraphConfig)

# Defining the nodes of the graph
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# Setting the entrypoint as `agent`
workflow.set_entry_point("agent")

# Adding conditional edge 
workflow.add_conditional_edges(
   # Starting node
   "agent",
   # Passing the function which will define the next node to call
   should_continue,
   {
       # If `tools`, then call the tool node.
       "continue": "action",
       # else finish.
       "end": END,
   },
)

# Adding an edge from `tools` to `agent`.
workflow.add_edge("action", "agent")

# Compiling the entire flow to create a LangChain runnable 
graph = workflow.compile()

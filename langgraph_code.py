# Import the necesary modules
import os
import requests
import json
import functools
import operator
import time
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START
from typing import Annotated, Sequence
from langchain_core.tools import tool
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv


# Define the workflow prompt for the various usecases
USECASE = "What is the state of venture capital for AI in 2024? Provide a summary of the key trends and investments in the AI sector."

# Load environment variables
load_dotenv()

# Set OPENAI KEY AND MODEL from environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

OPENAI_MODEL = "gpt-4o"

# Set the environment variables for web automation
EMERGENCE_API_KEY = os.getenv("EMERGENCE_API_KEY")
URL = "https://api.emergence.ai/v0/orchestrators/em-web-automation/workflows"


def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards completing the task."
                " If you or any of the other assistants have the final deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " If you can't do the task using the tools, you can stop."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)


def get_api_response(
    base_url: str, method: str, headers: dict, payload: dict = {}
) -> dict:
    """
    Sends an HTTP request to a specified URL using the given method, headers, and payload.

    Parameters:
    - base_url (str): The URL for the API endpoint.
    - method (str): The HTTP method to use (e.g., 'GET' or 'POST').
    - headers (dict): The headers for the request.
    - payload (dict): The data to be sent with the request.

    Returns:
    - dict: The final response from the API in JSON format.
    """

    # Create a http request for the given API endpoint
    response = requests.request(method, base_url, headers=headers, json=payload)
    response = json.loads(response.text)

    return response


# Create the tool for the web automation
@tool()
def web_automation_tool(prompt: str) -> str:
    """A tool that can take a high-level natural language task as a prompt and break it down into multiple web navigation steps to accomplish the task, and perform those steps in a web browser. This tool can only do web navigation steps.

    Parameters:
    prompt (str): The  or prompt to guide the web navigation task.

    Returns:
    str: Relevant information retrieved from the web navigation results.

    """
    try:
        # Define the base URL for the API endpoint
        base_url = URL

        # Create the request payload with the  prompt
        payload = {
            "prompt": prompt,
        }

        # Set headers with content type and API key for authorization
        headers = {
            "Content-Type": "application/json",
            "apikey": EMERGENCE_API_KEY,
        }

        # Parse the response to extract the workflow ID for tracking status
        response = get_api_response(
            base_url=base_url, method="POST", headers=headers, payload=payload
        )
        workflowId = response["workflowId"]

        # Construct the URL to check the status of the workflow
        base_url = f"{URL}/{workflowId}"

        # Empty payload for the GET request to check status
        payload = {}

        # Set headers with content type and API key for authorization
        headers = {
            "apikey": EMERGENCE_API_KEY,
        }

        response = get_api_response(base_url=base_url, method="GET", headers=headers)

        print(response)

        # loop: Continue checking until the workflow status is "SUCCESS"
        while response["data"]["status"] in ["IN_PROGRESS", "QUEUED", "PLANNING"]:
            response = get_api_response(
                base_url=base_url, method="GET", headers=headers, payload=payload
            )
            time.sleep(10)

        # Check workflow status for the current workflow ID
        if (
            response["data"]["workflowId"] == workflowId
            and response["data"]["status"] == "SUCCESS"
        ):
            # Return the result if the workflow completes successfully
            return response["data"]["output"]["result"]

        # Return error message if the workflow does not complete successfully
        return "An error occurred while getting result of the prompt."
    except Exception as e:
        return f"An error occurred while performing the web automation: {str(e)}"


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


# Helper function to create a node for an agent, processing its state and appending it to the global state
def agent_node(state, agent, name):
    """
    Create a processing node for an agent within the workflow.

    Parameters:
    - state: Current state of the agent, including messages and sender info
    - agent: The agent being processed
    - name: Name of the agent for tracking

    Returns:
    - A dictionary with updated messages and sender information.
    """
    # Invoke the agent with the given state
    result = agent.invoke(state)

    # We convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "sender": name,
    }


# Initialize the language model
llm = ChatOpenAI(model=OPENAI_MODEL)

# Create the web automation agent with a system prompt to retrieve web information
web_automation_agent = create_agent(
    llm,
    [web_automation_tool],
    system_message="You are an expert in web automation. You can take a high-level natural language task as a prompt and break it down into multiple web navigation steps to accomplish the task, and perform those steps in a web browser. You could only do web navigation steps and nothing more.",
)

# Set up a node specifically for the webautomation agent
web_automation_node = functools.partial(
    agent_node, agent=web_automation_agent, name="web_automation"
)

# Define the tools for the workflow
tools = [web_automation_tool]

# Node for handling tool-specific actions
tool_node = ToolNode(tools)


# Define a router function to control workflow routing logic
def router(state):
    """
    Router to direct the workflow based on the last message and its content.

    Parameters:
    - state: Current state with messages and sender info.

    Returns:
    - Routing directive ('continue', 'call_tool', or END).
    """
    messages = state["messages"]
    last_message = messages[-1]  # Get the last message in the state

    # Route to tool execution if a tool was invoked
    if last_message.tool_calls:
        return "call_tool"

    # If the final answer is found in the content, stop the workflow
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return END

    return "continue"  # Continue the workflow if neither condition is met


# Initialize the workflow with the AgentState structure
workflow = StateGraph(AgentState)

# Define and add nodes for each agent and the tool handler
workflow.add_node("web_automation", web_automation_node)
workflow.add_node("call_tool", tool_node)
workflow.add_conditional_edges(
    "web_automation",
    router,
    {"continue": "web_automation", "call_tool": "call_tool", END: END},
)


# Special handling for the tool call routing
workflow.add_conditional_edges(
    "call_tool",
    # Use the sender field to route back to the agent who originally called the tool
    lambda x: x["sender"],
    {
        "web_automation": "web_automation",
    },
)

# Start the workflow by setting an edge from START to the initial websearch agent
workflow.add_edge(START, "web_automation")

# Compile the workflow into an executable graph structure
graph = workflow.compile()

# Run the graph's event stream with initial input
events = graph.stream(
    {
        "messages": [HumanMessage(content=USECASE)],
    },
    # Maximum number of steps to take in the graph
    {"recursion_limit": 150},
)


# Iterate through each event generated by the graph to produce the output
try:
    for event in events:
        print(event)
        print("----")
except Exception as e:
    print(f"An error occurred while streaming the event: {str(e)}")

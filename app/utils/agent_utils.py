from langchain_core.messages import AIMessage, ToolMessage
from app.utils.agents.bridge_agent import create_deposit_agent

AGENT_REGISTRY = {}


def get_last_ai_message(response_data):
    """
    Extracts the content of the last AIMessage from the response data.
    """
    messages = response_data.get("messages", [])

    # Iterate in reverse to find the last AI message
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:  # Ensure it's an AI message with content
            return message.content

    return "No valid AI response found."


def get_last_message(message):
    response_payload = {
        "ai_message": "None",
        "tool_response": "None"
    }
    if isinstance(message, AIMessage) and message.content:
        response_payload = {
            "ai_message": message.content,
            "tool_response": "None"
        }
    if isinstance(message, ToolMessage) and message.content:
        response_payload = {
            "ai_message": "None",
            "tool_response": message.content
        }
    return response_payload


def get_relevant_tool_message(response_data):
    """
    Extracts the last ToolMessage before the last AIMessage.
    If the last AIMessage doesn't have a preceding ToolMessage, find the most recent one before it.
    """
    messages = response_data.get("messages", [])

    last_ai_index = None
    last_tool_message = None

    # Iterate in reverse to locate the last AIMessage
    for i in range(len(messages) - 1, -1, -1):
        message = messages[i]

        if isinstance(message, AIMessage):
            if last_ai_index is None:
                last_ai_index = i  # Store the index of the last AIMessage
            else:
                break  # Stop when encountering an earlier AIMessage

        if isinstance(message, ToolMessage):
            last_tool_message = message.content  # Store the most recent ToolMessage

        # If we found an AIMessage and already stored a ToolMessage before it, return it
        if last_ai_index is not None and last_tool_message:
            return last_tool_message

    return "None"


def load_agents_on_startup():
    """
    Loads agents from the database into memory on startup.
    """
    zec_graph = create_deposit_agent()
    AGENT_REGISTRY["zecIntentAgent"] = {"graph": zec_graph}

import json
from typing import Dict
from langchain_core.messages import SystemMessage
from langchain.tools import Tool
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from app.utils.openai_utils import llm
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient
from app.config import MONGODB_URI
import requests


def create_deposit_agent():
    BRIDGE_CHAINED_USER = "https://bridge.chaindefuser.com/rpc"

    def deposit_zec(*args):
        print("Args: ", args)
        wallet_address = args[0]
        print("Params: ", wallet_address)
        rpc_request = {
            "id": "dontcare",
            "jsonrpc": "2.0",
            "method": "deposit_address",
            "params": [{"account_id": wallet_address, "chain": "zec:mainnet"}]
        }

        response = requests.post(BRIDGE_CHAINED_USER, json=rpc_request)
        if response.ok:
            data = response.json().get("result", [])
            return {
                "address": data["address"],
                "chain": data["chain"],
                "type": "deposit",
                "success": True
            }
        else:
            return {
                "type": "deposit",
                "success": False
            }

    deposit_zec_tool = Tool(
        name="deposit_zec",
        func=deposit_zec,
        description="This tool is to create zec deposit address based on the near wallet address which user is giving. "
                    "require filed is address. NEAR Protocol, addresses can be either human-readable account IDs (like "
                    "jane.near) or implicit addresses (64 characters, like fb9243ce...), both representing the same "
                    "public key"
                    "- `address` must be a string.\n"
                    "Call this tool only when all parameters are correctly provided."
    )

    # ✅ Bind tools to LLM
    llm_with_tools = llm.bind_tools([deposit_zec_tool])

    # ✅ Define Assistant Node
    sys_msg = SystemMessage(
        content="You are an AI-powered Trading Agent. You are capable of doing deposit, investment and trading on Zcash"
                "network"
    )

    def assistant(state: MessagesState):
        response = llm_with_tools.invoke([sys_msg] + state["messages"])
        return {"messages": state["messages"] + [response]}

    # ✅ Build Graph
    builder = StateGraph(MessagesState)
    builder.add_node("tools", ToolNode([deposit_zec_tool]))
    builder.add_node("assistant", assistant)
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)

    mongodb_client = MongoClient(MONGODB_URI)
    checkpointer = MongoDBSaver(
        mongodb_client,
        db_name="new_memory",
        checkpoint_ns="AGY"
    )

    graph = builder.compile(checkpointer=checkpointer)
    return graph

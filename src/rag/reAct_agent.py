"""
ReAct agent setup for document retrieval and question answering.
"""

import os

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

from src.config.settings import Config
from src.llms.groq import llm
from src.rag.retriever_setup import get_retriever

config = Config()

# Initialize tools
tools = [get_retriever()]

# Load document description if available
if os.path.exists("description.txt"):
    with open("description.txt", "r", encoding="utf-8") as f:
        description = f.read()
else:
    description = None

# Create agent prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the retriever tool to search and answer questions based on the uploaded documents."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])


def get_agent_executor() -> AgentExecutor:
    """Dynamically create an AgentExecutor with the latest retriever tool."""
    tools = [get_retriever()]
    react_agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=react_agent,
        tools=tools,
        handle_parsing_errors=True,
        max_iterations=2,
        verbose=True,
        return_intermediate_steps=True
    )


# Backward compatibility
agent_executor = get_agent_executor()

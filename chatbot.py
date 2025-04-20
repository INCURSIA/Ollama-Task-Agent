from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from tools import get_datetime,delete_file_action,youtube_search_opener
import json
import os,re
import webbrowser

# Load your local Ollama model
model = ChatOllama(model="llama3.2")  # Change version if needed

tools = [get_datetime, delete_file_action,youtube_search_opener]

template = """You are a helpful assistant who can perform actions using tools.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: think about what to do. If the input require some file make sure u get the file name and pass it to tools.
if the input is related to watching smth run or open youtube only once no need to rerun the tools open best suitable youtube query.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action. Havee it as "" if none provided

Observation: the result of the action
... (you can repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: your final response to the user

Begin!

Question: {input}
{agent_scratchpad}
"""


prompt = PromptTemplate.from_template(template)

# ✅ Create agent
agent = create_react_agent(llm=model, tools=tools, prompt=prompt)
agent_with_tools = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True)

# LangGraph setup
workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    result = agent_with_tools.invoke({"input": state["messages"][-1].content})
    if not isinstance(result, AIMessage):
        result = AIMessage(content=str(result))
    return {"messages": state["messages"] + [result]}

workflow.add_node("model", call_model)
workflow.set_entry_point("model")
app = workflow.compile()

# Memory Handling
memory_file = "memory.json"

def load_conversation():
    try:
        with open(memory_file, "r") as f:
            raw = json.load(f)
            return [
                HumanMessage(m["content"]) if m["type"] == "human"
                else AIMessage(m["content"]) if m["type"] == "ai"
                else SystemMessage(m["content"])
                for m in raw
            ]
    except FileNotFoundError:
        return []

def save_conversation(messages):
    raw = [{"type": msg.type, "content": msg.content} for msg in messages]
    with open(memory_file, "w") as f:
        json.dump(raw, f)

# Main Chat Function
def chat(query: str) -> str:
    messages = load_conversation()
    messages.append(HumanMessage(query))
    output = app.invoke({"messages": messages})
    save_conversation(output["messages"])
    last_msg = output["messages"][-1].content

    final_part = last_msg.split("Final Answer:")[-1]
    urls = re.findall(r"https://www\.youtube\.com/results\?search_query=[\w+%-]+", final_part)

    if urls:
        webbrowser.open(urls[-1])  # Open only the final clean one

    return last_msg


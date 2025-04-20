# --- Imports ---
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import ollama, base64, json, os, re, webbrowser

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent

# --- Tools ---
from tools import get_datetime, delete_file_action, youtube_search_opener
tools = [get_datetime, delete_file_action, youtube_search_opener]

# --- Vision Functions ---
def blip_caption(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def llava_reasoning(image_path, question="What is happening in this image?"):
    with open(image_path, "rb") as img_file:
        img_b64 = base64.b64encode(img_file.read()).decode("utf-8")
    response = ollama.chat(
        model="llava",
        messages=[{"role": "user", "content": question, "images": [img_b64]}]
    )
    return response["message"]["content"]

# --- LLM Agent Setup ---
model = ChatOllama(model="llama3.2")  # Use local Ollama model

template = """You are a helpful assistant who can use tools to perform tasks.

Available tools:
{tools}

FORMAT INSTRUCTIONS:
Use the following format:

Question: the input question
Thought: reason through what needs to be done
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action, in quotes (e.g., "filename.png" or "search term")
Observation: result of the action
... (Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: [your final response to the user]

RULES:
- Only call a tool if needed.
- Do NOT perform more than 3 action steps total. Think before using tools.
- If input refers to a file, pass the file name to the tool.
- Never break the format. Every action must have its matching Action Input and Observation.
- Do not include raw URLs in Action Input unless specifically asked.

Begin!

Question: {input}
{agent_scratchpad}
"""


prompt = PromptTemplate.from_template(template)
agent = create_react_agent(llm=model, tools=tools, prompt=prompt)
agent_with_tools = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- LangGraph Setup ---
workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    result = agent_with_tools.invoke({"input": state["messages"][-1].content})
    if not isinstance(result, AIMessage):
        result = AIMessage(content=str(result))
    return {"messages": state["messages"] + [result]}

workflow.add_node("model", call_model)
workflow.set_entry_point("model")
app = workflow.compile()

# --- Conversation Memory ---
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

# --- Master Chat + Vision Interface ---
def multimodal_chat(input_text=None, image_path=None):
    vision_response = ""
    if image_path:
        caption = blip_caption(image_path)
        reasoning = llava_reasoning(image_path, f"What can you infer about this scene: '{caption}'?")
        vision_response = f"\nImage Caption: {caption}\nVision Insight: {reasoning}\n"

    full_input = (vision_response + input_text) if input_text else vision_response
    messages = load_conversation()
    messages.append(HumanMessage(full_input))

    output = app.invoke({"messages": messages})
    save_conversation(output["messages"])
    last_msg = output["messages"][-1].content

    # Optional: Auto-open YouTube
    urls = re.findall(r"https://www\.youtube\.com/results\?search_query=[^\"\s]+", last_msg)
    if urls:
        print("Opening YouTube link:", urls[-1])
        webbrowser.open(urls[-1])

    return last_msg

# response = multimodal_chat(
#     input_text="Based on this image, suggest a YouTube tutorial.",
#     image_path = r"images\hey.jpeg")
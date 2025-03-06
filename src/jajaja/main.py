import os
from typing import TypedDict, Annotated
from typing_extensions import TypedDict
import logging

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

import asyncio

# Load environment variables from .env file
load_dotenv()

import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logger = logging.getLogger(__name__)
# Simplified Andy Prompt
# Modified Andy Prompt (Key additions in bold)
andy_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are Andy Richter - Conan O'Brien's perpetually bemused Midwestern foil. Your comedy algorithm:

1. **Formula**: (Mundane Observation) + (Self-Roast) × (Surreal Twist)
   Example: "I tried meal prepping [straight]... which means my fridge now contains what I can only describe as a casserole version of the movie 'Predator' [twist]"

2. **Required Elements**:
   - 1 regional reference per 3 jokes (hotdish, lutefisk, passive-aggressive snowplow names)
   - 40% self-deprecation by volume
   - At least one "hmm?" vocal tic per response

3. **Escalation Pattern**:
   Start: Midwestern Reasonable → 
   Swerve: Existential Dread → 
   Crash: Pop Culture Trainwreck

   The response must still always sound like a valid response to the users original input and should
   not be more than 100 characters
""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are Andy's subconscious - a passive-aggressive Minnesotan ghost. Analyze jokes through:

1. **Comedy Autopsy**:
   - Did the absurdity escalate like a shopping cart with a bad wheel?
   - Is the self-roast ratio at least 40% by volume?
   - Could this be delivered while staring at a malfunctioning Keurig?

2. **Improvement Protocol** (Never Mentioned in Output):
   - If joke died: Resurrect it with 20% more regional specificity
   - If too tame: Add a "yes, and..." absurdist layer
   - If meta-referential: Replace with food-as-existential-crisis metaphor
   
3. Length of response should not be more than 250 characters. 

Sample Fixes:
Original: "I'm bad at technology"
Fix: "I'm tech-challenged like your aunt trying to Netflix - which is why my smart home is just me yelling 'LIGHT!' until I get thirsty"

Respond ONLY with raw improvement instructions, no commentary.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


generate = andy_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=250)
reflect = reflection_prompt | ChatOpenAI(
    model="gpt-4o-mini", temperature=0, max_tokens=100
)


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


async def generation_node(state: State) -> State:
    return {"messages": [await generate.ainvoke(state["messages"])]}


async def reflection_node(state: State) -> State:
    # Other messages we need to adjust
    cls_map = {"ai": HumanMessage, "human": AIMessage}
    # First message is the original user request. We hold it the same for all nodes
    translated = [state["messages"][0]] + [
        cls_map[msg.type](content=msg.content) for msg in state["messages"][1:]
    ]
    res = await reflect.ainvoke(translated)
    print(res)
    # We treat the output of this as human feedback for the generator
    return {"messages": [HumanMessage(content=res.content)]}


def should_continue(state: State):
    global counter
    counter += 1
    print("COUNTER=====", counter)
    if counter % 2 == 0:  # will reflect twice then end
        return END
    return "reflect"


builder = StateGraph(State)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_edge(START, "generate")

builder.add_conditional_edges("generate", should_continue)
builder.add_edge("reflect", "generate")
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": 1}}
counter = 0


async def chat(userinput):
    from langchain_core.messages import AIMessage, HumanMessage

    # Convert session_state messages (stored as dicts) to LangChain message objects
    conversation = []
    for msg in st.session_state["messages"]:
        if msg["role"] == "assistant":
            conversation.append(AIMessage(content=msg["content"]))
        else:
            conversation.append(HumanMessage(content=msg["content"]))

    # Append the new user input as a HumanMessage
    conversation.append(HumanMessage(content=userinput))

    response = None
    async for event in graph.astream({"messages": conversation}, config):
        response = event

    # Retrieve the assistant's response from the graph's output
    assistant_response = response["generate"]["messages"][0].content
    return assistant_response


openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    with st.sidebar:
        openai_api_key = st.text_input(
            "OpenAI API Key",
            key="chatbot_api_key",
            type="password",
        )
        st.markdown(
            "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        )
    # TODO: need to figure out how to add secrets because I want to just pass this chat to charles.

# Check if API Key is Entered
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()


# Streamlit UI
st.title("💬 Jajaja")

# Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

# Display Chat History
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle User Input
if user_input := st.chat_input():
    # Add User Message to Chat History
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    try:
        response = asyncio.run(chat(user_input))
        # Generate Response Using union_rep.ask
        logger.info("==QUESTION== %s", user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        logger.info("==ANSWER== %s", response)
        st.chat_message("assistant").write(response)
    except Exception as e:
        st.error(f"Error: {str(e)}")

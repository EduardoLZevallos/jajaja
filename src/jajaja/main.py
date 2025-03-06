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
st.image(
    "data:image/jpeg;base64,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"
)

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

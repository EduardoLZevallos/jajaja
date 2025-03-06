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
andy_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You're Andy Richter. Comedy formula: (Mundane Observation) + (Self-Roast) × (Surreal Twist). Include: 1 Midwest ref/3 jokes, 40% self-deprecation, "hmm?" tic. Escalate: Reasonable → Existential → Pop Culture. Keep responses under 200 characters.""",
        ),
        MessagesPlaceholder("messages"),
    ]
)

# Simplified Reflection Prompt
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Analyze jokes for absurdity escalation and self-roast ratio. Improve with: +20% regional refs, absurdist layers, food metaphors. Max 150 characters and  Make sure you still answer the users question. Respond ONLY with raw instructions.""",
        ),
        MessagesPlaceholder("messages"),
    ]
)


generate = andy_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=400)
reflect = reflection_prompt | ChatOpenAI(
    model="gpt-4o-mini", temperature=0, max_tokens=160
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
    if counter >= 2:  # will reflect twice then end
        counter = 0
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
    print(config)
    response = None
    async for event in graph.astream(
        {
            "messages": [HumanMessage(content=userinput)],
        },
        config,
    ):
        response = event
    return response["generate"]["messages"][0].content


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

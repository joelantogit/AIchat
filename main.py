"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv


from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory



load_dotenv()

def load_agent():
    chat = OpenAI(temperature=0.9)
    tools = load_tools(["serpapi"])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = initialize_agent(tools,chat,  agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True, memory=memory)
    return agent

agent = load_agent()

def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = ChatOpenAI(temperature=0.1)
    tool_names = ["serpapi"]
    tools = load_tools(tool_names)
    # agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
    chain = ConversationChain(llm=llm)
    return chain

# chain = load_chain()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


# def get_text():
#     input_text = st.text_input("You: ", "Hello, how are you?", key="input")
#     return input_text
#

user_input = st.text_input("chat with super chatgpt5")

if user_input:
    output = agent.run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")


#%%

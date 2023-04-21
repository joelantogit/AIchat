"""Python file to serve as the frontend"""
import streamlit as st
from dotenv import load_dotenv
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import PromptTemplate
from langchain.agents import (initialize_agent, AgentType, Tool, load_tools)
from langchain.memory import ConversationBufferMemory


load_dotenv()

llm = OpenAI(temperature=0.1)
chat = ChatOpenAI(temperature=0.1)
memory = ConversationBufferMemory(memory_key="chat_history")
general_query_prompt = PromptTemplate(template="{query}", input_variables=["query"])
general_query = LLMChain(llm=chat, prompt=general_query_prompt)

general_query_tool = Tool(
    name="Language Model",
    func=general_query.run,
    description="use this tool for general queries, conversation and reasoning"
)

tools = load_tools(['serpapi'])
# tools[0].name='Intermediate Answer'
# tools[0].description="use this as a search tool to find answers for unknown questions"
# tools.append(general_query_tool)

tools.append(general_query_tool)
agent = initialize_agent(
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    verbose=True,
    max_iteration=3,
    memory=memory
)

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Super Assistant", page_icon=":robot:")
st.header("Super Assistant")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

user_input = st.text_input("say hi!")

if user_input:
    output = agent.run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

# %%

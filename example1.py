import os

from constants import openai_key

from langchain.llms import Ollama
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory

# os.environ["OPENAI_API_KEY"] = openai_key
import streamlit as st


st.title("Celebrity Search Engine")

input_text = st.text_area("search the topic u want")

# PROMPT TEMPLATE EXAMPLE 1
first_input_template = PromptTemplate(
    input_variables=["name"],
    template="Tell me about {name}.",
)

# message buffer
person_memory = ConversationBufferMemory(input_key="name", memory_key="chat_hsitory")
dob_memory = ConversationBufferMemory(input_key="person", memory_key="chat_hsitory")
descrp_memory = ConversationBufferMemory(
    input_key="dob", memory_key="description_history"
)

# LLAMA LLM MODEL
llm = Ollama(base_url="http://bore.zubairmh.xyz:11434", model="zephyr", temperature=0.8)
chain = LLMChain(
    llm=llm,
    prompt=first_input_template,
    verbose=True,
    output_key="person",
    memory=person_memory,
)


# PROMPT TEMPLATE EXAMPLE 2
second_input_template = PromptTemplate(
    input_variables=["person"],
    template="when was {person} born ",
)
chain2 = LLMChain(
    llm=llm,
    prompt=second_input_template,
    verbose=True,
    output_key="dob",
    memory=dob_memory,
)

# PROMPT TEMPLATE EXAMPLE 2
third_input_template = PromptTemplate(
    input_variables=["dob"],
    template=" Mention 5 major events that happened around {dob} in the world.",
)
chain3 = LLMChain(
    llm=llm,
    prompt=third_input_template,
    verbose=True,
    output_key="description",
    memory=descrp_memory,
)

# making a parent chain that runss both the chains in sequence

parent_chain = SequentialChain(
    chains=[chain, chain2, chain3],
    input_variables=["name"],
    output_variables=["person", "dob", "description"],
    verbose=True,
)
if input_text:
    st.write(parent_chain({"name": input_text}))

    with st.expander("Person name"):
        st.info(person_memory.buffer)

    with st.expander("Major Events"):
        st.info(descrp_memory.buffer)

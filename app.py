import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()
mistral_api_key = os.getenv("MISTRAL_API_KEY")
llm = ChatMistralAI(model="mistral-small-latest", api_key=mistral_api_key, rate_limiter = None)

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="search")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Use arxiv, wikipedia and duckduckgo search tool for information based on user queries.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

tools = [search, arxiv, wiki]
search_agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=search_agent, tools=tools, verbose=True, handle_parsing_errors=True)

st.title("Search Engine")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi I am a chatbot that can search for information on the web. How can I help you today?"}
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt:= st.chat_input(placeholder="What is your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False, collapse_completed_thoughts = False)
        response = agent_executor.invoke(
            {"input": prompt}, {"callbacks": [st_cb]}
        )
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response["output"])   
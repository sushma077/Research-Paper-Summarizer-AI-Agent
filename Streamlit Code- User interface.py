from langgraph.graph import StateGraph, START, END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict, Dict, List
import requests
import streamlit as st
import os


# Function to initialize the LLM with API key
def initialize_llm(api_key: str):
    # Set the API key in environment variables
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
    
    llm = HuggingFaceEndpoint(
        endpoint_url="openai/gpt-oss-20b",
        task="text-generation",
        huggingfacehub_api_token=api_key
    )
    
    return ChatHuggingFace(llm=llm)

# Defining State 
class PaperInfo(TypedDict):
    prompt: str
    topic: List[str]
    top_search: int
    title: List[str]
    abstract: List[str]
    url: List[str]
    citationCount: List[int]
    result: str
    model: object  # Added to store the model instance
    
# Function to get papers from Semantic Scholar API
def get_papers(Info: PaperInfo) -> PaperInfo:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"

    for i in range(Info['top_search']):
        params = {
            "query": Info['topic'][i],
            "fields": "title,url,abstract,citationCount",
            "limit": 1,
            "offset": 0
        }
        response = requests.get(url, params=params) 
        data = response.json()

        for paper in data.get("data", []):
         Info["abstract"].append(paper.get("abstract"))
         Info["title"].append(paper.get("title"))
         Info["url"].append(paper.get("url"))    
         Info["citationCount"].append(paper.get("citationCount"))

    return Info

# Final answer Drafting function
def draft_answer(Info: PaperInfo) -> PaperInfo:
    prompt = f"Using the following papers, draft the summarization of each paper '{Info['topic']}'.\n\n"
    for i in range(len(Info['title'])):
        prompt += f"Title: {Info['title'][i]}\nAbstract: {Info['abstract'][i]}\nURL: {Info['url'][i]}\nCitations: {Info['citationCount'][i]}\n\n"
    prompt += "Summarize the key points from these papers in a concise manner.\n\n Title: \n Citations: \n Abstract Summary: \n URL: \n "
    
    response = Info['model'].invoke(prompt)
    Info['result'] = response.content
    return Info

# Generating Paper Titles
def generate_titles(Info: PaperInfo) -> PaperInfo:
    prompt = Info['prompt']
    response = Info['model'].invoke(prompt)
    titles = response.content.split('\n')
    Info['topic'] = [title.strip() for title in titles if title.strip()]
    return Info

# Graph Definition
graph = StateGraph(PaperInfo)

# adding nodes
graph.add_node("generate_titles", generate_titles)
graph.add_node("get_papers", get_papers)
graph.add_node("draft_answer", draft_answer)

# adding edges
graph.add_edge(START, "generate_titles")
graph.add_edge("generate_titles", "get_papers")
graph.add_edge("get_papers", "draft_answer")
graph.add_edge("draft_answer", END)

# Completing the graph
research_paper_graph = graph.compile()

# Creating Streamlit App
st.title("Get Research Papers and Summarize")

# API Key input
api_key = st.text_input("Enter your Hugging Face API Key:", type="password", help="Get your API key from https://huggingface.co/settings/tokens")

input_topic = st.text_input("Enter the research topic:")
top_search = st.number_input("Number of top papers to fetch:", min_value=1, max_value=10, value=3)

if st.button("Get Papers"):
    if not api_key:
        st.error("Please enter your Hugging Face API key.")
    elif not input_topic:
        st.error("Please enter a research topic.")
    else:
        try:
            # Initialize the model with the provided API key
            model = initialize_llm(api_key)
            
            prompt = f"User gave us {input_topic} as a topic. We need to find relevant research papers for this topics. Understand the topic and give me top {top_search} research paper's title. Make sure the titles are relevant to the topic. all topic should give a sequential learning to user. for example, if topic is: 'Linear Regression' and top_search is 3 then 1st paper should be the 1st foundational paper 2nd should be with further sequential papers which improved it further and same with 3rd. Give me only the titles in the response and each title should be in new line."
            
            initial_info: PaperInfo = {
                "prompt": prompt,
                "topic": [],
                "top_search": top_search,
                "title": [],
                "abstract": [],
                "url": [],
                "citationCount": [],
                "result": "",
                "model": model
            }
            
            with st.spinner("Searching for papers and generating summary..."):
                result = research_paper_graph.invoke(initial_info)
            
            st.subheader("Summarized Result:")
            st.write(result['result'])
            
        except Exception as e:
            st.error(f"Error initializing the model: {str(e)}. Please check your API key.")
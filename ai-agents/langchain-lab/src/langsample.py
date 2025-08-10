import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
# from langchain_community.llms import GPT4All
# from langchain_ollama import OllamaLLM

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model="gemini-2.5-flash-lite"

# Initialize Gemini Pro model
llm = ChatGoogleGenerativeAI(
    model=model,
    google_api_key=api_key,
    temperature=0.7
)
# llm = GPT4All(
#     model="C:/Users/kanna/AppData/Local/nomic.ai/GPT4All/qwen2.5-coder-7b-instruct-q4_0.gguf"
# )

# llm = OllamaLLM(model="mistral")

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a comprehensive analysis about {topic}"
)

chain = LLMChain(llm=llm, prompt=prompt)

# Initialize conversation with memory
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Define tools for the agent
tools = [
    Tool(
        name="Analysis Chain",
        func=chain.run,
        description="Useful for when you need to analyze a topic in detail. Input should be a topic string."
    ),
    Tool(
        name="Conversation",
        func=conversation.run,
        description="Useful for having a conversation and remembering context. Input should be a message string."
    )
]

# Reinitialize the agent with tools
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Example of running the agent
if __name__ == "__main__":
    try:
        response = agent.run("Analyze the impact of artificial intelligence on healthcare and summarize our discussion.")
        print(response)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
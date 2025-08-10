# from langchain_community.llms import GPT4All
from langchain_ollama import OllamaLLM

# llm = GPT4All(
#     model="C:/Users/kanna/AppData/Local/nomic.ai/GPT4All/qwen2.5-coder-7b-instruct-q4_0.gguf"
# )
llm = OllamaLLM(model="mistral")
response = llm.invoke("The first man on the moon was?")
print(response)
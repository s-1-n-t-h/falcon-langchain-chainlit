import os
import chainlit as cl
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from dotenv import load_dotenv
from langsmith import Client
from langchain.callbacks.tracers.langchain import wait_for_all_tracers

client = Client()

model_id = 'tiiuae/falcon-7b-instruct'

falcon_llm = HuggingFaceHub(huggingfacehub_api_token=os.getenv('HF_API_KEY'),
                            repo_id=model_id,
                            model_kwargs={"temperature":0.8,"max_new_tokens":2000})


template = """

You are an AI assistant that provides helpful answers to user queries.

{question}

"""


@cl.on_message
async def main(message: str):

    prompt = PromptTemplate(template=template, input_variables=['question'])
    falcon_chain = LLMChain(llm=falcon_llm,
                        prompt=prompt,
                        verbose=True)
    #response = falcon_chain.run(message)
    wait_for_all_tracers()
    await cl.Message(
        content=falcon_chain.run(message)
    ).send()



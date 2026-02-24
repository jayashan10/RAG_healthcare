import os
import openai  # Commented out OpenAI
from dotenv import load_dotenv
from typing import List
import modal
from modal import Cls
from langsmith import Client
from langsmith.run_helpers import traceable

# Optional replicate import (may fail on Python 3.14+)
try:
    import replicate
except Exception as e:
    print(f"Warning: Could not import replicate: {e}")
    replicate = None

# Initialize LangSmith client
client = Client()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Clinical-Trial-Assistant"

# Set your OpenAI API key
load_dotenv("config.env")
# openai.api_key = os.getenv('OPENAI_API_KEY')  # Commented out OpenAI


# Create a client instance (optional - will be None if no API key)
client = None
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        client = openai.OpenAI()
except Exception as e:
    print(f"Warning: Could not initialize OpenAI client: {e}")


@traceable(run_type="chain")
def generate_response(chunks: List[dict], query: str, chat_history: List[str]) -> dict:
    if client is None:
        return {
            "answer": "Error: OpenAI API not configured. Please use the Llama or Mistral model instead.",
            "sources": [],
        }

    # Combine the most relevant chunks, limiting to a reasonable token count
    context = " ".join(
        [chunk["text"] for chunk in chunks[:5]]
    )  # Adjust based on your chunk sizes and model's context window

    # Prepare the messages list
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an AI assistant specializing in clinical trials. Your role is to provide accurate, helpful, and concise information based on the given context. If the answer is not contained within the context, say that you don't have enough information to answer accurately.",
                }
            ],
        }
    ]

    # Add chat history to messages
    for i, message in enumerate(chat_history[-5:]):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": [{"type": "text", "text": message}]})

    # Add user query to messages
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Context: {context}, Query: {query}"}
            ],
        }
    )

    # Generate a response using OpenAI's model
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=1,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    answer = response.choices[0].message.content.strip()

    # Prepare sources information
    sources = [
        {
            "document": chunk["metadata"]["document"],
            "page_number": chunk["metadata"]["page_number"],
            "excerpt": chunk["text"][:100] + "...",
        }
        for chunk in chunks[:5]
    ]

    return {"answer": answer, "sources": sources}


@traceable(run_type="chain")
def generate_response_modal_llama(
    chunks: List[dict], query: str, chat_history: List[str]
) -> dict:
    context = " ".join(
        [chunk["text"] for chunk in chunks[:5]]
    )  # Adjust based on your chunk sizes and model's context window

    # Prepare the prompt
    system_message = "You are an AI assistant specializing in clinical trials. Your role is to provide accurate, helpful, and concise information based on the given context. If the answer is not contained within the context, say that you don't have enough information to answer accurately."

    chat_history_formatted = "\n".join(
        [
            f"Human: {msg}" if i % 2 == 0 else f"Assistant: {msg}"
            for i, msg in enumerate(chat_history[-5:])
        ]
    )

    prompt = f"{system_message}\n\nContext: {context}\n\nChat History:\n{chat_history_formatted}\n\nHuman: {query}\n\nAssistant:"

    # Prepare the input for the Modal-deployed model
    model_input = {"prompts": [prompt]}

    # Call the Modal-deployed model
    Model = Cls.from_name("TensorRT-LLaMa", "Model")
    model = Model()  # Instantiate the class
    response = model.generate.remote(**model_input)

    # The response is a list with one item (since we sent one prompt)
    answer = response[0] if response else "No response generated."

    # Prepare sources information
    sources = [
        {
            "document": chunk["metadata"]["document"],
            "page_number": chunk["metadata"]["page_number"],
            "excerpt": chunk["text"][:100] + "...",  # First 100 characters of the chunk
            # 'score': chunk['score']
        }
        for chunk in chunks[:5]  # Include sources for top 3 chunks
    ]

    return {"answer": answer, "sources": sources}


@traceable(run_type="chain")
def generate_response_mistral(
    chunks: List[dict], query: str, chat_history: List[str]
) -> dict:
    if replicate is None:
        return {
            "answer": "Error: Replicate API not available. Please use the Llama model instead.",
            "sources": [],
        }

    # Combine relevant chunks
    context = " ".join([chunk["text"] for chunk in chunks[:5]])

    # Format chat history
    chat_history_formatted = "\n".join(
        [
            f"Human: {msg}" if i % 2 == 0 else f"Assistant: {msg}"
            for i, msg in enumerate(chat_history[-5:])
        ]
    )

    # Prepare the system prompt and full message
    system_message = """You are an AI assistant specializing in clinical trials. 
    Your role is to provide accurate, helpful, and concise information based on the given context. 
    If the answer is not contained within the context, say that you don't have enough information to answer accurately."""

    prompt = f"{system_message}\n\nContext: {context}\n\nChat History:\n{chat_history_formatted}\n\nHuman: {query}\n\nAssistant:"

    # Call Mistral model via Replicate
    output = replicate.run(
        "mistralai/mistral-7b-v0.1",
        input={
            "top_k": 0,
            "top_p": 0.95,
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.7,
            "length_penalty": 1,
            "max_new_tokens": 150,
            "prompt_template": "{prompt}",
            "presence_penalty": 0,
            "log_performance_metrics": False,
        },
    )

    # Replicate returns an iterator, collect the response
    answer = "".join([chunk for chunk in output])

    # Prepare sources information
    sources = [
        {
            "document": chunk["metadata"]["document"],
            "page_number": chunk["metadata"]["page_number"],
            "excerpt": chunk["text"][:100] + "...",
        }
        for chunk in chunks[:5]
    ]

    return {"answer": answer, "sources": sources}

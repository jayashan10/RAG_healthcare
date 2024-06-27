import os
import llamaindex
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

"""Stores all secret keys"""

from dotenv import load_dotenv
load_dotenv()

import os

class Config:
    GEOCLIENT_V2_API_KEY = os.getenv("GEOCLIENT_V2_API_KEY")
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
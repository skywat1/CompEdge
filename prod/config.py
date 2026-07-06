"""Stores all secret keys"""

from dotenv import load_dotenv
load_dotenv()

import os

class Config:
    GEOCLIENT_V2_API_KEY = os.getenv("GEOCLIENT_V2_API_KEY")
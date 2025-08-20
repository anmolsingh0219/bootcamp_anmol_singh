from dotenv import load_dotenv
import os

def load_env():
    load_dotenv()

def get_key(key_name):
    return os.getenv(key_name)
import os
import base64
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# --- 1. MLI Model setup ---
MLI_URL = os.getenv("MLI_URL")
MLI_LLM_MODEL = os.getenv("MLI_LLM_MODEL")
MLI_AUTH_TOKEN = os.getenv("MLI_AUTH_TOKEN")

if MLI_URL and MLI_AUTH_TOKEN:
    mli_model = ChatOpenAI(
        base_url=MLI_URL,
        api_key=MLI_AUTH_TOKEN,
        model=MLI_LLM_MODEL,
        http_client=httpx.Client(verify=False)
    )
else:
    mli_model = None

# --- 2. PPD Model setup ---
PPD_URL = os.getenv("PPD_URL")
PPD_LLM_MODEL = os.getenv("PPD_LLM_MODEL")
PPD_LLM_USER = os.getenv("PPD_LLM_USER")
PPD_LLM_PASSWORD = os.getenv("PPD_LLM_PASSWORD")

if PPD_URL and PPD_LLM_USER and PPD_LLM_PASSWORD:
    auth_token = base64.b64encode(f"{PPD_LLM_USER}:{PPD_LLM_PASSWORD}".encode("utf-8")).decode("utf-8")
    llm_headers = {"Authorization": f"Basic {auth_token}"}
    ppd_model = ChatOpenAI(
        base_url=PPD_URL,
        api_key=auth_token,
        model=PPD_LLM_MODEL,
        default_headers=llm_headers,
        http_client=httpx.Client(verify=False)
    )
else:
    ppd_model = None


def get_model(name: str = "mli"):
    """
    Returns the configured chat model by name.
    Supported names: 'mli', 'ppd'.
    """
    if name == "mli":
        if mli_model is None:
            raise ValueError("MLI model environment variables are not fully configured.")
        return mli_model
    elif name == "ppd":
        if ppd_model is None:
            raise ValueError("PPD model environment variables are not fully configured.")
        return ppd_model
    raise ValueError(f"Unknown model name: {name}. Use 'mli' or 'ppd'.")

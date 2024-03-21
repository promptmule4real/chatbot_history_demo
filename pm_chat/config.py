class Config:
    # API URLs
    BASE_URL = "https://api.promptmule.com"
    LOGIN_URL = f"{BASE_URL}/login"
    API_KEYS_URL = f"{BASE_URL}/api-keys"
    PROMPT_URL = f"{BASE_URL}/prompt"

    # Default headers
    DEFAULT_HEADERS = {
        "Content-Type": "application/json",
    }

    # Default parameter values
    DEFAULT_MODEL = "gpt-3.5-turbo"
    DEFAULT_MAX_TOKENS = 100
    DEFAULT_TEMPERATURE = 1.0
    DEFAULT_SEMANTIC = "1.0"
    DEFAULT_SEM_NUM = "10"
    DEFAULT_API = "openai"
    USERNAME = 'demo'
    PASSWORD = 'Abcd@1234!'

    # Other configuration values
    USE_LOCAL_API = False
    API_TIMEOUT = 30
    # Add more configuration values as needed
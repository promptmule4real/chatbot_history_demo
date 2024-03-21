# config_loader.py
from dotenv import load_dotenv
import os
import logging
#from pm_chat.setup_logging import logger 
#import sys  # Import sys to access stdout

#logger = logging.getLogger()
#logger.setLevel(logging.DEBUG)

# # Create a StreamHandler for stdout
# stdout_handler = logging.StreamHandler(sys.stdout)
# stdout_handler.setLevel(logging.DEBUG)  # Set the log level for this handler

# # Optionally, set a formatter for the handler
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# stdout_handler.setFormatter(formatter)

# # Add the handler to the logger
# logger.addHandler(stdout_handler)

load_dotenv()

def load_env_vars(var_name):
    
    if var_name == "OPENAI_API_KEY":
        return os.getenv("OPENAI_API_KEY", False)
    elif var_name == "ANTHROPIC_API_KEY":
        return os.getenv("ANTHROPIC_API_KEY", False)
    elif var_name == "GOOGLE_PROJECT_ID":
        return os.getenv("GOOGLE_PROJECT_ID", False)
    elif var_name == "GOOGLE_LOCATION_ID":
        return os.getenv("GOOGLE_LOCATION_ID", False)
    elif var_name == "PROMPTMULE_API_KEY":
        return os.getenv("PROMPTMULE_API_KEY", False)
    elif var_name == "TEMPERATURE": 
        return os.getenv("TEMPERATURE", 1.0)
    elif var_name == "MAX_TOKENS":
        return os.getenv("MAX_TOKENS", 100)
    elif var_name == "USE_VERTEX_AI":
        return os.getenv("USE_VERTEX_AI", False)
    elif var_name == "USE_ANTRHOPIC":
        return os.getenv("USE_ANTRHOPIC", False)
    elif var_name == "USE_LOCAL_LLM":
        return os.getenv("USE_LOCAL_LLM", False)
    elif var_name == "USE_PROMTPMULE":
        return os.getenv("USE_PROMTPMULE", False)
    elif var_name == "USE_LOCAL_PROMTPMULE":
        return os.getenv("USE_LOCAL_PROMTPMULE", False)
    elif var_name == "USE_OPENAI":
        return os.getenv("USE_OPENAI", False)
    elif var_name == "TEST_PRINT_RESPONSES":
        print("Env TEST_PRINT_RESPONSES:", os.getenv("TEST_PRINT_RESPONSES"))
        logger.debug("Testing printout? %s", os.getenv("TEST_PRINT_RESPONSES"))
        return os.getenv("TEST_PRINT_RESPONSES", False)
    else:    
        return False 
# pm_chat/__init__.py

# Import main functionality into the package namespace for easy access
# from .chatbot import start_chatbot  # Uncomment if chatbot.py is moved inside pm_chat and you want to expose it
from .api_clients import *  # Changed to relative import
#from .api2_clients import *  # Changed to relative import

# from .utils import *        # Changed to relative import
from .config_loader import *  # Changed to relative import

# from .chatbot_ui import *
from .chatbot_ui_config import *
from .setup_logging import *

# Initialize any package-wide variables or configurations
# config = load_env_vars()  # Ensure load_env_vars() is designed to work as expected

# Optionally, define what should be available with "from pm_chat import *"
__all__ = ["setup_logging", "PromptMuleClient", "load_env_vars"]

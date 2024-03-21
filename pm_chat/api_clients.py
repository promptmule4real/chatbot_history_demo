"""
File: api_clients.py
Author: [Your Name]
Date: [Current Date]
Description:
   This file contains the API client classes for interacting with various APIs used in the chat application.
   The main client is the PromptMuleClient, which handles authentication, API key management, and making requests
   to the PromptMule API for generating text, fetching similar responses, and retrieving prompt history.

   The file also includes placeholder imports for other API clients such as Google's Vertex AI and OpenAI, which
   can be integrated into the application as needed.

Classes:
   - PromptMuleClient: The main client class for interacting with the PromptMule API.

Methods:
   - PromptMuleClient:
       - __init__: Initializes the PromptMuleClient with default configurations.
       - login_to_promptmule: Logs in to the PromptMule API using the provided username and password.
       - list_api_keys: Lists all API keys associated with the logged-in user's account.
       - set_api_key: Sets the client's API key based on the provided identifier (app name or index number).
       - extract_response: Extracts response details from the API response JSON.
       - extract_contents_and_cache_flags: Extracts content strings and cache flags from the API response JSON.
       - generate_text: Generates text using the PromptMule API with the specified parameters.
       - fetch_responses_based_on_similar_prompts: Fetches similar responses based on a user's prompt using the PromptMule API.
       - fetch_prompt_and_response_history_with_dates: Retrieves prompt history based on a given date range, number of prompts, and optional sorting parameters.
       - get_user_stats: Retrieves user statistics and usage data for all API keys associated with the logged-in user.
       - fetch_api_keys: Fetches the list of API keys for the logged-in user.
       - fetch_usage_stats: Fetches usage stats for a specific API key.
       - create_auth_headers: Helper function to create authorization headers for API requests.

Dependencies:
   - requests: Used for making HTTP requests to the PromptMule API.
   - logging: Used for logging messages and errors.
   - json: Used for parsing JSON responses from the API.

Usage:
   1. Create an instance of the PromptMuleClient.
   2. Log in to the PromptMule API using the login_to_promptmule method.
   3. Set the API key using the set_api_key method.
   4. Use the various methods provided by the PromptMuleClient to interact with the PromptMule API, such as generating text,
      fetching similar responses, retrieving prompt history, and getting user statistics.
   5. Integrate other API clients (e.g., Google's Vertex AI, OpenAI) as needed for additional functionality.
"""

import requests  # for PromptMule calls
import logging  # to keep us honest
import json
from pm_chat.setup_logging import logger

requests_log = logging.getLogger("requests.packages.urllib3")
requests_log.setLevel(logging.DEBUG)
requests_log.propagate = True


# PromptMule API client
class PromptMuleClient:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.url = "https://api.promptmule.com"
        self.token = None
        self.headers = {
            "Content-Type": "application/json",
        }
        self.auth_hearders = f"Bearer {self.token},"
        self.use_local_api = False  # Set to True if using a local testing environment
        self.api_key = None
        self.api_key_list = {}
        self.username = None
        self.password = None
        self.app_name = None

    def login_to_promptmule(self, username=None, password=None):

        print("Logging in to PromptMule")
        print(f"Username: {username}")
        print(f"Password: {password}")

        if username is None or password is None:
            self.logger.error("Login failed. Missing username or password.")
            return False

        self.username = username
        self.password = password
        login_url = f"{self.url}/login"
        headers = self.headers
        data = {"username": username, "password": password}
        try:
            print("Attempting login to PromptMule")
            response = requests.post(url=login_url, json=data, headers=headers)
            response.raise_for_status()  # Raises an exception for 4XX/5XX errors

            response_data = response.json()
            if response_data.get("success"):
                self.token = response_data["token"]
                print("Successfully logged in to PromptMule")
                return True
            else:
                self.logger.error(f"Login failed: {response_data.get('message')}")
                return (
                    None  # Consider returning None or raising an exception on failure
                )
        except requests.RequestException as e:
            self.logger.error(f"Error during login: {e}")
            return None  # Consider returning None or raising a custom exception

    def list_api_keys(self, username, password):
        """Lists all API keys associated with the logged-in user's account."""
        if not self.token:
            if username is not None and password is not None:
                self.login_to_promptmule(username, password)
                self.logger.error("Cannot list API keys without a valid token.")
            else:
                self.logger.error("Cannot list API keys without a valid token.")
                return {"success": False, "message": "Authentication required."}

        list_keys_url = f"{self.url}/api-keys"
        headers = self.create_auth_headers()

        try:
            response = requests.get(url=list_keys_url, headers=headers)
            response.raise_for_status()
            api_keys_data = response.json()

            if "api-keys" in api_keys_data and api_keys_data["api-keys"]:
                print(
                    f"Successfully retrieved {len(api_keys_data['api-keys'])} API keys."
                )
                self.api_key_list = api_keys_data
                return {"success": True, "api_keys": api_keys_data["api-keys"]}
            else:
                return {"success": False, "message": "No API keys found for the user."}
        except requests.HTTPError as e:
            self.logger.error(f"HTTP error while listing API keys: {e}")
            return {"success": False, "message": f"HTTP error: {e}"}
        except Exception as e:
            self.logger.error(f"Unexpected error while listing API keys: {e}")
            return {"success": False, "message": f"Unexpected error: {e}"}

    def set_api_key(self, identifier, first_login=False):
        """
        Sets the client's API key based on the provided identifier, which can be an app name (str) or an index number (int).

        Parameters:
        - identifier (str|int): The name of the app or the index of the app to retrieve the API key for.
        - first_login (bool): Indicates if this is the first login attempt, triggering a login and API key listing.

        Returns:
        - bool: True if the API key is successfully set, False otherwise.
        """
        print("Inside set_api_key\n")
        print("API key list retrieved:", json.dumps(self.api_key_list))

        # Validate prerequisites for API key listing
        if self.username is None or self.password is None:
            self.logger.error("Username or password not set.")
            return False
        print(f"Setting API key for identifier: {identifier}")

        # Attempt to log in and list API keys if not done already or explicitly requested
        if (
            first_login
            or self.api_key in [None, ""]
            or "api-keys" not in self.api_key_list
        ):
            login_success = (
                self.login_to_promptmule(self.username, self.password)
                if first_login or self.token in [None, ""]
                else True
            )
            if not login_success:
                self.logger.error(
                    "Login failed or not attempted due to missing credentials."
                )
                return False
            self.list_api_keys(self.username, self.password)

        print("API key list retrieved:", json.dumps(self.api_key_list))

        # Ensure API keys are properly listed
        if "api-keys" not in self.api_key_list or not self.api_key_list["api-keys"]:
            self.logger.error("Failed to list API keys or no API keys found.")
            return False

        api_keys = self.api_key_list["api-keys"]
        print("api_key retrieved:", api_keys)

        if isinstance(identifier, int):
            # identifier is treated as app_index
            if not 0 <= identifier < len(api_keys):
                self.logger.error(
                    f"App Index {identifier} is out of bounds for the API key list."
                )
                return False
            self.api_key = api_keys[identifier].get("api-key")
        elif isinstance(identifier, str):
            # identifier is treated as app_name
            for key_info in api_keys:
                if key_info.get("app-name") == identifier:
                    self.api_key = key_info.get("api-key")
                    break
            else:
                self.logger.warning(f"API key for app name '{identifier}' not found.")
                return False

        if self.api_key:
            print("API key set successfully.")
            return True
        else:
            self.logger.error("Failed to set API key.")
            return False

    def extract_response(self, response_json):
        details = []

        for choice in response_json.get("choices", []):
            # Extract various details from each choice
            detail = {
                "content": choice.get("message", {}).get("content", ""),
                "is_cached": choice.get("is_cached", False),
                "prompt": response_json.get("prompt", ""),
                "prompt_id": choice.get("prompt-id", ""),
                "score": choice.get("score", None),
                "finish_reason": choice.get("finish_reason", ""),
                "index": choice.get("index", None),
            }
            details.append(detail)

        return details

    def extract_contents_and_cache_flags(self, response_json):
        """
        Extracts content strings and cache flags from a response JSON.

        Parameters:
        - response_json (dict): The JSON response containing choices.

        Returns:
        - tuple: A tuple containing two lists, one for contents and another for cache flags.
        """
        try:
            choices = response_json.get("choices", [])
            contents = [
                choice.get("message", {}).get("content", "") for choice in choices
            ]
            cache_flags = [choice.get("is_cached", False) for choice in choices]

            return contents, cache_flags
        except Exception as e:
            self.logger.error(f"Error extracting content and cache flags: {e}")
            return [], []  # Return empty lists as a safe fallback

    def generate_text(
        self,
        contents,
        model="gpt-3.5-turbo",
        max_tokens=100,
        temperature=1.0,
        semantic="0.1",
        sem_num="10",
        api="openai",
    ):
        """
        Generates text using the PromptMule API.

        Parameters:
            contents (str): The content or prompt for the model.
            model (str): The specific model to use for generation.
            max_tokens (int): The maximum number of tokens to generate.
            temperature (float): The temperature value for generation.
            semantic (str): The semantic similarity score to use.
            sem_num (str): The number of semantic matches to return.
            api (str): The API to use for generation, default is "openai".

        Returns:
            dict: The JSON response from the API, or None in case of error.
        """
        prompt_url = f"{self.url}/prompt"
        headers = self.headers  # Assuming self.headers is a dict
        headers["x-api-key"] = self.api_key

        if contents is None or contents == " ":
            contents = "Respond with the word 'Hello PromptMule'"

        data = {
            "model": model,
            "messages": [{"role": "user", "content": contents}],
            "max_tokens": max_tokens,
            "api": api,
            "temperature": temperature,
            "semantic": str(semantic),
            "sem_num": sem_num,
        }

        try:
            if self.use_local_api:
                self.logger.info("Using LOCAL PromptMule instance")
                # Placeholder for local API call logic. Implement as needed.
                # response = self.promptmule_local_api_call(data)
            else:
                self.logger.info("Using PromptMule API")
                logger.info(
                    f"In generate_text() Sending request to {prompt_url}\n with headers: {headers}\n and data: {data}\n"
                )
                # logger.info(f"max_tokens: {max_tokens}, temperature: {temperature}, semantic: {semantic}, sem_num: {sem_num}, api: {api}")
                logger.info(
                    f"max_tokens: {type(max_tokens)}, temperature: {type(temperature)}, semantic: {type(semantic)}, sem_num: {type(sem_num)}, api: {type(api)}"
                )
                print("sending json: ", json.dumps(data, indent=4))
                response = requests.post(url=prompt_url, json=data, headers=headers)
                logger.info(f"Response json: {response.json()}\n")
                print("Response json: ", json.dumps(response.json(), indent=4))
                response.raise_for_status()  # Raises HTTPError for bad responses
                return response.json()  # Returns the parsed JSON response
        except requests.HTTPError as http_err:
            self.logger.error(f"HTTP error occurred: {http_err}")
        except Exception as e:
            self.logger.error(f"Error: {e}")

        return response.json()  # Return None in case of any errors

    def fetch_responses_based_on_similar_prompts(
        self,
        contents,
        model="gpt-3.5-turbo",
        max_tokens=100,
        temperature=1.0,
        semantic=0.1,
        sem_num="10",
        api="openai",
    ):
        """
        Fetches similar responses based on a user's prompt using the PromptMule API. This function is a specialized
        use case of the generate_text function, configured to fetch responses with a focus on semantic similarity.

        Parameters:
        - contents (str): The user's input prompt.
        - model (str): The model to use for comparison. Default is "gpt-3.5-turbo".
        - max_tokens (int): The maximum number of tokens to generate.
        - temperature (float): The temperature value for generation.
        - semantic (str): The semantic similarity score 'percentage' to determine which responses are returned. Default is '0.1'.
        - sem_num (str): The number of semantic matches to return. Default is '10'.
        - api (str): The API to use for generation, default is "openai".

        Returns:
        - dict: A dictionary containing similar prompts or an error message.
        """
        print(f"Fetching similar responses based on prompt: {contents}")
        return self.generate_text(
            contents, model, max_tokens, temperature, semantic, sem_num, api
        )

    def fetch_prompt_and_response_history_with_dates(
        self,
        formatted_start_date,
        formatted_end_date,
        num_prompts=10,
        is_cached=True,
        sortOrder="desc",
        sortBy=None,
    ):
        """
        Retrieves prompt history based on a given date range, number of prompts, and optional sorting parameters.
        """
        # Log the action for debugging and monitoring
        print(
            f"Fetching prompt history from PromptMule. Start Date: {formatted_start_date}, End Date: {formatted_end_date}, "
            f"Num Prompts: {num_prompts}, Cached: {is_cached}, Sort Order: {sortOrder}, Sort By: {sortBy}"
        )

        # Define the endpoint for fetching prompt history
        logger.info(
            f"\nFetching prompt history from PromptMule. Start Date: {formatted_start_date}, End Date: {formatted_end_date}\n"
        )
        print(
            f"Fetching prompt history from PromptMule. Start Date: {formatted_start_date}, End Date: {formatted_end_date}\n"
        )
        endpoint = f"{self.url}/prompt"  # Corrected endpoint as per API structure

        # Setup the query parameters
        # - `start-date` (date, required): Start date for the date range (YYYY-MM-DD format).
        # - `end-date` (date, required): End date for the date range (YYYY-MM-DD format).
        # - `is_cached` (boolean, optional): Filter for cached prompts (`true` or `false`).
        # - `limit` (integer, optional, default=10): Maximum number of prompts to retrieve.
        # - `sortOrder` (string, optional, default=`desc`): Sorting order for the results (`desc` or `asc`).
        # - `sortBy` (string, optional, default=`created` the creation date): Field to sort the results by (e.g., `tokens`, `is_cached`, `latency`).
        params = {
            "start-date": formatted_start_date,
            "end-date": formatted_end_date,
            "limit": num_prompts,
            "is_cached": is_cached,
            "sortOrder": sortOrder,  # can be 'asc' or 'desc'
            "sortBy": sortBy,  # can be 'latency', 'is_cached', '
        }

        headers = self.headers
        headers["x-api-key"] = self.api_key
        headers["Authorization"] = f"Bearer {self.token}"
        print(f"Headers: {headers}")
        print(f"Params: {params}")
        print(f"Endpoint: {endpoint}")
        try:
            # Make the GET request to the PromptMule API
            response = requests.get(endpoint, headers=self.headers, params=params)
            print("Response: ", response)
            # Check for HTTP errors
            response.raise_for_status()

            # Parse the JSON response
            data = response.json()

            # Check if 'prompts' key exists in response data
            if "prompts" in data:
                return data["prompts"]
            else:
                self.logger.error(
                    "Unexpected response structure: no 'prompts' key found"
                )
                return []
        except requests.HTTPError as http_err:
            self.logger.error(f"HTTP error occurred: {http_err}")
            return []
        except Exception as e:
            self.logger.error(f"General error retrieving prompt history: {e}")
            return []

    def get_user_stats(self):
        print("Retrieving user stats")
        if not self.token:
            self.logger.error(
                "Token is missing. Authentication required for get_user_stats."
            )
            return {"error": "Authentication required."}

        try:
            # Fetch the list of API keys
            api_keys = self.fetch_api_keys()
            if not api_keys:
                print("No API keys found for this user.")
                return []

            # Fetch and compile usage data for each API key
            usage_data = [self.fetch_usage_stats(api_key) for api_key in api_keys]
            usage_data = [
                data for data in usage_data if data
            ]  # Filter out any None values

            print("Successfully retrieved usage data for all API keys.")
            return usage_data
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching user stats: {e}")
            return {"error": "Failed to fetch user stats."}

    def fetch_api_keys(self):
        """Fetches the list of API keys for the logged-in user."""
        list_keys_url = f"{self.url}/api-keys"
        headers = self.create_auth_headers()
        response = requests.get(url=list_keys_url, headers=headers)
        response.raise_for_status()  # May raise an HTTPError for bad responses

        api_keys_data = response.json()
        return api_keys_data.get("api-keys", [])

    def fetch_usage_stats(self, api_key_info):
        """Fetches usage stats for a specific API key."""
        usage_url = f"{self.url}/usage"
        headers = self.create_auth_headers(api_key_info["api-key"])
        response = requests.get(url=usage_url, headers=headers)
        response.raise_for_status()  # May raise an HTTPError for bad responses

        usage_info = response.json()
        return {**api_key_info, **usage_info}

    def create_auth_headers(self, api_key=None):
        """Helper function to create authorization headers."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        if api_key:
            headers["x-api-key"] = api_key
        return headers

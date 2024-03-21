# PromptMule Chat Application

This is a chat application built using the PromptMule API and the `pm_chat` libraries. It provides a user-friendly interface for interacting with a chatbot powered by the PromptMule API.

## Features

- Chatbot interface for engaging in conversations with the PromptMule AI
- Search and retrieval of similar responses based on user prompts
- Prompt and response history management
- User authentication and profile management
- Integration with the PromptMule API for generating responses
- Token usage analysis and savings tracking
- Data export functionality for prompt history and similarity data

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/promptmule4real/chatbot_history_demo.git
   ```

2. Navigate to the project directory:
   ```
   cd chatbot_history_demo
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up the necessary environment variables:
   - Create a `.env` file in the project root.
   - Add the following variables to the `.env` file:
     ```
     PROMPTMULE_PASSWORD=your_promptmule_password
     PROMPTMULE_USERNAME=your_promptmule_username
     or
     PROMPTMULE_API_KEY=your_promptmule_api_key
     ```

5. Run the application:
   ```
   python chatbot_v7.py
   or
   run.sh
   ```

6. Access the application in your web browser at `http://localhost:5006`.

## Usage

1. Login to the application using your PromptMule credentials.
2. Use the chatbot interface to engage in conversations with the PromptMule AI.
3. Utilize the search functionality to find similar responses based on user prompts.
4. Explore the prompt and response history to review past interactions.
5. Manage your user profile and preferences.
6. Analyze token usage and savings through visualizations.
7. Export prompt history and similarity data as CSV files.

## API Clients

The application utilizes the `api_clients.py` module to interact with the PromptMule API. This module provides the necessary functionality to authenticate, retrieve prompts, generate responses, and manage API keys.

### PromptMuleClient

The `PromptMuleClient` class is the main client for interacting with the PromptMule API. It provides methods for authentication, API key management, prompt retrieval, and response generation.

#### Key Methods

- `login_to_promptmule(username, password)`: Logs in to the PromptMule API using the provided username and password.
- `list_api_keys(username, password)`: Lists all API keys associated with the logged-in user's account.
- `set_api_key(identifier, first_login)`: Sets the client's API key based on the provided identifier (app name or index number).
- `generate_text(contents, model, max_tokens, temperature, semantic, sem_num, api)`: Generates text using the PromptMule API with the specified parameters.
- `fetch_responses_based_on_similar_prompts(contents, model, max_tokens, temperature, semantic, sem_num, api)`: Fetches similar responses based on a user's prompt using the PromptMule API.
- `fetch_prompt_and_response_history_with_dates(formatted_start_date, formatted_end_date, num_prompts, is_cached, sortOrder, sortBy)`: Retrieves prompt history based on a given date range, number of prompts, and optional sorting parameters.

### Other Clients

The application also includes clients for interacting with other APIs such as Google Cloud and OpenAI. These clients are used for specific functionalities within the application.

## Contributing

Contributions to the PromptMule Chat Application are welcome! If you find any bugs, have feature requests, or want to contribute enhancements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- The PromptMule API for providing the underlying chatbot functionality.
- The open-source libraries and frameworks used in this project.

## Contact

For any questions or inquiries, please contact [info@promptmule.com](mailto:info@promptmule.com).
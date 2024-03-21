"""
Title: PromptMule Chat Application
Author: [Your Name]
Date: [Current Date]
Description: A chat application built using the Panel library and the PromptMule API. The application provides a user interface
            for interacting with a chatbot, searching for similar responses, managing prompt and response history, and analyzing
            user statistics. It includes features such as chat interaction, prompt searching, data export, user authentication,
            and data visualization.

Main Components:
- Imports and Setup
- Global Variables
- PromptMule Client
- UI Components
- Event Handlers
- Chat Interface
- Layout and Tabs
- Data Visualization
- User Profile
- Main Layout and Servable

Dependencies:
- Panel
- Pandas
- Bokeh
- PromptMule API (pm_chat)
- scikit-learn

Usage:
1. Set up the necessary environment variables (PROMPTMULE_PASSWORD, PROMPTMULE_USERNAME) in a .env file.
2. Run the script to launch the application.
3. Access the application through the provided URL.
4. Interact with the chatbot, search for similar responses, manage prompt history, and explore user statistics.
"""
import os
import pandas as pd
import panel as pn
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time
from collections import Counter
from pm_chat import PromptMuleClient
from pm_chat.setup_logging import logger
from bokeh.models import FactorRange
from bokeh.plotting import figure
from bokeh.models import FactorRange
import numpy as np
from typing import Tuple, List, Dict, Callable

# Load environment variables at the start
load_dotenv()

# Initialize Panel with the Material Design template and custom styles.
pn.extension(
    template="material"
)  # Ensure materialize.css is a path to your custom CSS if needed
pn.extension("tabulator")

# Global DataFrame for prompt data
df = pd.DataFrame()  # prompt history table
ef = pd.DataFrame()  # prompt similarity table

# Configure the Panel state and raw CSS for styling
pn.state.template.param.update(title="Prompt Library & Chatbot Demo")
pn.config.raw_css.append(
    """
    /* Custom CSS styles */
    pane_background {"background-color": "#FFFFFF"},
    prompt_display_margin: {"margin": "10px 10px 10px 10px"},
    layout_background: {"background": "white"},
    layout_margin: {"margin": "10px"},

    .material-pane {
        border-radius: 4px;  /* Rounded corners */
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);  /* Elevation effect */
        padding: 16px;  /* Consistent padding */
        background-color: #FFFFFF;  /* Light background */
        height: 100%;  /* Full height */
    }

    .material-label {
        font-weight: bold;
        margin-bottom: 16px;
    }

    .material-select {
        margin-bottom: 16px;
    }

    .material-row {
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 10px;
    }

    .scrollable-prompt-pane {
        max-height: auto; /* Fixed maximum height */
        white-space: normal; /* Allow text wrapping */
    }

    .prompt-search-box input {
        width: 60%; /* Make the input field responsive */
        box-sizing: border-box; /* Include padding and border in the element's total width */
    }

    .fixed-width-prompt-history {
        width: 35%; /* Fixed width for prompt history layout */
        overflow-y: auto; /* Vertical scroll for overflow */
        overflow-x: auto; /* Hide horizontal scroll */
        max-height: auto; /* Fixed maximum height */
    }

    .responsive-datetime-picker .bk-input-group {
        width: auto; /* Ensure the widget fills the container */
        box-sizing: border-box; /* Include padding and border in the element's total width */
    }
    
    .material-slider .bk-slider {
        background-color: white;
        height: 6px;
        border-radius: 3px;
    }

    .material-slider .bk-slider-handle {
        background-color: #2196F3;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        top: -5px;
    }

    .material-slider .bk-slider-handle:hover {
        background-color: #1976D2;
    }

    }    
"""
)


# Define custom CSS for smaller font and cell padding
tabulator_css = """
.pn-tabulator .tabulator-cell {
    font-size: 12px;  # Adjust the font size as needed
    white-space: normal;
    word-wrap: break-word;
    padding: 4px;  # Adjust the cell padding as needed
}
"""

# Apply the CSS to Panel (this adds the CSS to the entire document, affecting all Tabulator widgets)
pn.config.raw_css.append(tabulator_css)

# Initialize the PromptMule client for API calls
promptmule_client = PromptMuleClient()
PROMPTMULE_PASSWORD = os.getenv("PROMPTMULE_PASSWORD")
PROMPTMULE_USERNAME = os.getenv("PROMPTMULE_USERNAME")

promptmule_client.login_to_promptmule(
    username=PROMPTMULE_USERNAME, password=PROMPTMULE_PASSWORD
)
promptmule_client.set_api_key(identifier=0, first_login=True)

# Use the Markdown component for displaying counts as it offers more flexibility in formatting
retrieved_prompt_count_display = pn.pane.Markdown("")

# Logger setup
logger.debug("In Chatbot UI...\n")

# Setup for datetime range picker
default_start_date = datetime.now() - timedelta(days=60)
default_end_date = datetime.now()

datetime_range_picker = pn.widgets.DatetimeRangePicker(
    name="Prompt Creation Date (GMT)",
    value=(default_start_date, default_end_date),
    css_classes=["responsive-datetime-picker"],
)


# Define a custom formatter function to wrap cell content
def wrap_formatter(cell, formatterParams, onRendered):
    cellValue = cell.getValue()
    return '<div style="white-space:normal; word-wrap: break-word;">{}</div>'.format(
        cellValue
    )


response_similarity_table = pn.widgets.Tabulator(
    value=df,
    height=700,
    pagination="local",
    page_size=10,
    selectable="checkbox",
    header_filters=True,
    layout="fit_columns",
    sizing_mode="stretch_width",  # Add this line to make the table responsive
    header_align="left",  # Align the header text to the left
    theme="materialize",  # Apply the 'midnight' theme
    formatters={"*": wrap_formatter},  # Apply the custom formatter to all columns
)


# Setup for the Tabulator widget
prompt_history_table = pn.widgets.Tabulator(
    value=df,
    height=700,
    pagination="local",
    page_size=10,
    selectable="checkbox",
    header_filters=True,
    layout="fit_columns",
    sizing_mode="stretch_width",  # Add this line to make the table responsive
    header_align="left",  # Align the header text to the left
    theme="materialize",  # Apply the 'midnight' theme
    formatters={"*": wrap_formatter},  # Apply the custom formatter to all columns
)


# Function to update the prompt history table
def update_prompt_history_table(prompt_listing):
    global df  # Ensure df is recognized as the DataFrame you're updating
    # Handle empty prompt listings
    if not prompt_listing:
        prompt_listing = [
            {"request-time": "N/A", "prompt": "No prompts found", "response": "N/A"}
        ]
    logger.info(f"Prompt Listing: {prompt_listing}")
    df = pd.DataFrame(prompt_listing)

    prompt_history_table.value = df  # Update the table


def update_response_similarity_table(response):
    logger.info(
        "In update_response_similarity_table: Updating the response similarity table."
    )
    global ef  # Ensure ef is recognized as the DataFrame you're updating

    if response and "choices" in response:
        flattened_choices = []
        for choice in response["choices"]:
            flattened_choice = {
                "prompt-id": choice.get("prompt-id", "N/A"),
                "content": choice.get("message", {}).get("content", "Content missing"),
                "score": choice.get("score", "Score missing"),
                "is_cached": choice.get("is_cached", "Caching status unknown"),
            }
            flattened_choices.append(flattened_choice)

        ef = pd.DataFrame(flattened_choices)
        response_similarity_table.value = ef
        return True
    else:
        logger.info("No similar prompts found or error occurred.")
        ef = pd.DataFrame()  # Clear ef if no data is present
        response_similarity_table.value = ef
        return False


def create_date_range_change_handler():
    """
    Creates a function to handle changes in the date range picker.
    Retrieves prompts from PromptMule based on the selected date range and quantity.
    Updates the prompt display pane with the retrieved prompts.
    """
    logger.info(
        "In create_date_range_change_handler: Retrieving prompts from PromptMule. Can I access global objects? quantity_select: {}".format(
            prompt_quantity_select.value
        )
    )

    def handle_date_range_change(event):
        # Extract new date range from the event
        if isinstance(event, dict):
            if "new" in event:
                start_date, end_date = event["new"]
            if "value" in event:
                start_date, end_date = event["value"]
        else:
            if hasattr(event, "new"):
                start_date, end_date = event.new
            else:
                start_date, end_date = event.value

        # Format dates for API call
        formatted_start_date = start_date.strftime("%Y-%m-%d")
        formatted_end_date = end_date.strftime("%Y-%m-%d")

        num_prompts = prompt_quantity_select.value

        logger.info(
            "Event sent Date Range: {} to {}\n".format(
                formatted_start_date, formatted_end_date
            )
        )

        logger.info(
            "Selected Date Range: {} to {} to retrieve {}\n".format(
                formatted_start_date, formatted_end_date, num_prompts
            )
        )

        prompts = promptmule_client.fetch_prompt_and_response_history_with_dates(
            formatted_start_date=formatted_start_date,
            formatted_end_date=formatted_end_date,
            num_prompts=num_prompts,
            sortOrder="desc",
            # sortBy="latency",  # None uses date default, or 'latency', 'tokens', 'is_cached' are only supported as of Feb 11, 2024
        )

        logger.info(f"Prompts Retrieved: {prompts}")

        # Format and display the fetched prompts in the UI.
        if not prompts:
            logger.info("No prompts retrieved.")
            update_prompt_history_table(
                [
                    {
                        "request-time": "N/A",
                        "prompt": "No prompts available for the selected quantity and date range.",
                        "response": "N/A",
                    }
                ]
            )
            return False
        else:
            update_prompt_history_table(prompts)
        return True

    return handle_date_range_change


def handle_quantity_change(event):
    """
    Fetches and displays prompts based on the selected quantity and current date range.
    This function is triggered by an event, such as changing the selection in a dropdown.
    """
    selected_quantity = event.new  # Extract the selected quantity from the event.

    # Retrieve the start and end dates from the datetime range picker UI component.
    start_date, end_date = datetime_range_picker.value
    formatted_start_date = start_date.strftime("%Y-%m-%d")
    formatted_end_date = end_date.strftime("%Y-%m-%d")

    logger.info(
        f"In handle_quantity_change: Fetching and displaying {selected_quantity} prompts."
    )

    # Use the PromptMule client to fetch prompts based on the specified criteria.
    prompts = promptmule_client.fetch_prompt_and_response_history_with_dates(
        formatted_start_date=formatted_start_date,
        formatted_end_date=formatted_end_date,
        num_prompts=selected_quantity,
    )

    logger.info(f"Prompts Retrieved: {prompts}")

    # Format and display the fetched prompts in the UI.
    if not prompts:
        logger.info("No prompts retrieved.")
        update_prompt_history_table(
            [
                {
                    "request-time": "N/A",
                    "prompt": "No prompts available for the selected quantity and date range.",
                    "response": "N/A",
                }
            ]
        )
    else:
        update_prompt_history_table(prompts)

    # Count how many times each value of a specific key (e.g., 'type') appears
    values = [prompt["type"] for prompt in prompts if "type" in prompt]
    counts = Counter(values)

    # Display the total count of retrieved prompts
    retrieved_prompt_count_display.object = (
        f"Ask: {selected_quantity} Retrieved: {len(prompts)}"
    )


async def handle_chat_interaction(
    contents: str, user: str, instance: pn.chat.ChatInterface
):
    """
    Callback function for handling chat interactions in the chat interface.

    Parameters:
    - contents (str): The content of the user's message.
    - user (str): The username of the user.
    - instance (pn.chat.ChatInterface): The instance of the chat interface.

    Returns:
    - None
    """
    # Initialize lists for responses and latencies
    instance.user = promptmule_client.username
    instance.avatar = promptmule_client.username
    instance.show_activity_dot = True

    responses = []
    latencies = []
    model_names = []
    model = "gpt-3.5-turbo"
    max_tokens = 100
    temperature = 0.2
    semantic_score = round(semantic_score_slider.value, 3)

    # PromptMule response
    model_names.append("PromptMule")
    if promptmule_client:
        logger.info(f"Prior to PromptMule call: {contents}")
        start_time = time.time()
        response = promptmule_client.generate_text(
            contents=contents,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            semantic=semantic_score,
            sem_num="1",
            api="openai",
        )
        promptmule_latency = time.time() - start_time
        promptmule_response = response["choices"][0]["message"]["content"]
        promptmule_cache = response["choices"][0]["is_cached"]

        logger.info(f"PromptMule Response: {promptmule_response}")
        logger.info(f"PromptMule Latency: {promptmule_latency}")
        logger.info(f"PromptMule Cache: {promptmule_cache}")
        responses.append(promptmule_response)
        latencies.append(promptmule_latency)
        instance.stream(
            promptmule_response
            + "\nlatency: "
            + f"{promptmule_latency:.3f}"
            + "\ncached: "
            + str(promptmule_cache),
            user=promptmule_client.username + f" @ PromptMule {model} ",
        )
    else:
        responses.append("")
        latencies.append(0)
        instance.stream(
            "PromptMule not configured correctly. Get support at https://app.promptmule.com",
            user="System",
            message=None,
        )

    display_latency_report(model_names, latencies, instance)


def display_latency_report(model_names, latencies, instance):
    """
    Displays the latency report for the models.
    """
    max_name_length = max(len(name) for name in model_names)
    table_rows = "\n".join(
        [
            f"{model:<{max_name_length}} was {latency:.3f}sec"
            for model, latency in zip(model_names, latencies)
        ]
    )
    instance.stream(table_rows, user="LLM Latency Report")


def login_to_promptmule(event):
    """
    Logs in to the PromptMule API using the provided username and password.
    """
    username = username_input.value
    password = password_input.value

    if username and password:
        promptmule_client.login_to_promptmule(username, password)
        promptmule_client.set_api_key(identifier=0, first_login=True)
        login_display.value = f"Logged in as {username}."
        logger.info("Logged in to PromptMule successfully.")
    else:
        logger.info("Username or password is missing.")
    return True


def logout_from_promptmule(event):
    """
    Logs in to the PromptMule API using the provided username and password.
    """
    promptmule_client.username = ""
    promptmule_client.password = ""
    promptmule_client.api_key = ""
    login_display.value = f"No one logged in."
    logger.info("Logged out from PromptMule successfully.")
    return True


def search_for_similar_prompts(event):
    """
    Searches for prompts similar to the user's input based on a similarity score.
    Updates the prompt display pane with the retrieved prompts.
    """

    search_text = prompt_search_box.value
    search_similarity = round(semantic_score_slider.value, 3)
    search_number = str(prompt_quantity_select.value)

    logger.info(
        f"Fetching similar prompts: {search_text}, Similarity: {search_similarity}, Number: {search_number}"
    )
    response = promptmule_client.generate_text(
        contents=search_text,
        semantic=round(semantic_score_slider.value, 3),
        sem_num=search_number,
    )

    print(f"In Response with Recent Prompts: {response}")

    # Check if response is successful and has 'choices'
    if "choices" in response:
        # Format the display string for each choice in the response
        prompt_count = len(response["choices"])
        update_response_similarity_table(response)
        retrieved_prompt_count_display.object = (
            f"Ask: {search_number} Retrieved: {prompt_count}"
        )
    else:
        logger.info("No similar prompts found or error occurred.")
        update_prompt_history_table(
            [{"request-time": "N/A", "prompt": "No prompts found", "response": "N/A"}]
        )
        retrieved_prompt_count_display.object = "Retrieved: 0"


def initialize_search_for_similar_prompts(event):
    """
    Searches for prompts similar to the user's input based on a similarity score.
    Updates the prompt display pane with the retrieved prompts.
    """
    logger.info("In initialize_search_for_similar_prompts: Fetching similar prompts.")
    search_text = " "
    search_similarity = round(semantic_score_slider.value, 3)
    search_number = str(prompt_quantity_select.value)

    logger.info(
        f"First Time Fetching similar prompts: {search_text}, Similarity: {search_similarity}, Number: {search_number}"
    )
    response = promptmule_client.generate_text(
        contents=search_text,
        semantic=round(semantic_score_slider.value, 3),
        sem_num=search_number,
    )
    # Check if response is successful and has 'choices'
    if "choices" in response:
        # Format the display string for each choice in the response
        prompt_count = len(response["choices"])
        update_response_similarity_table(response)
        retrieved_prompt_count_display.object = (
            f"Ask: {search_number} Retrieved: {prompt_count}"
        )
    else:
        logger.info("No similar prompts found or error occurred.")
        update_prompt_history_table(
            [{"request-time": "N/A", "prompt": "No prompts found", "response": "N/A"}]
        )
        retrieved_prompt_count_display.object = "Retrieved: 0"


# Function to export DataFrame to CSV, using the filename provided by the user
def export_df_to_csv(event):
    filename = filename_input.value
    if not df.empty:
        filepath = f"./{filename}.csv"
        df.to_csv(filepath, index=False)
        print(f"Data exported to '{filepath}'")
    else:
        print("DataFrame is empty. No data to export.")


# Function to export similarity DataFrame to CSV
def export_ef_to_csv(event):
    filename = filename_input.value
    if not ef.empty:
        filepath = f"./{filename}_similarity.csv"
        ef.to_csv(filepath, index=False)
        print(f"Similarity data exported to '{filepath}'")
    else:
        print("Similarity DataFrame is empty. No data to export.")


# Define text input for the filename
filename_input = pn.widgets.TextInput(
    name="Filename", value="exported_data", placeholder="Enter filename here"
)

# Define export buttons for df and ef DataFrames
export_df_button = pn.widgets.Button(
    name="Export Prompt History to CSV", button_type="primary"
)
export_df_button.on_click(export_df_to_csv)

export_ef_button = pn.widgets.Button(
    name="Export Similarity to CSV", button_type="primary"
)
export_ef_button.on_click(export_ef_to_csv)

# Define export controls for df and ef DataFrames
export_df_controls = pn.Row(filename_input, export_df_button)
export_ef_controls = pn.Row(filename_input, export_ef_button)

# Text input for the filename
filename_input = pn.widgets.TextInput(
    name="Filename", value="exported_data", placeholder="Enter filename here"
)


search_button = pn.widgets.Button(name="Search", button_type="primary")
search_button.on_click(
    lambda event: search_for_similar_prompts(prompt_search_box.value)
)

prompt_search_box = pn.widgets.TextInput(
    name="Similarity Search",
    placeholder="Discover similar responses...",
    css_classes=["prompt_search_box"],
)


# Initialize the chat interface with a callback function to handle chat interactions.
chat_interface = pn.chat.ChatInterface(
    callback=handle_chat_interaction,
    user=promptmule_client.username,
    active=True,
    show_undo=False,
    show_button_name=False,
    show_clear=False,
    width=800,
    sizing_mode="stretch_width",
)

# Send a welcome message via the chat interface on app start.
chat_interface.send(
    "Send a message to get a response from OpenAI via PromptMule. You can also search prompt history or look for similar responses.",
    user="System",
    respond=False,
)
import logging

logging.basicConfig(
    filename="reaction_changes.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)


# Create a function to log changes to reactions
def log_reaction_changes(event):
    with open("reaction_log.txt", "a") as f:
        f.write(f"Message ID: {event.model.id}, Reactions: {event.new}\n")


chat_message = pn.chat.ChatMessage(
    object="Hello world!", user="New User", reactions=["like", "heart"]
)
chat_message.param.watch(log_reaction_changes, "reactions")


prompt_quantity_select = pn.widgets.Select(
    name="Limit", options=[1, 10, 50, 100, 500], value=10, width=100
)

# Event listener for changes in the quantity select dropdown.
time.sleep(2)
prompt_quantity_select.param.watch(handle_quantity_change, "value")

# Setup the event handler for the datetime range picker
date_range_change_handler = create_date_range_change_handler()
datetime_range_picker.param.watch(
    lambda event: date_range_change_handler(event), "value"
)

# Trigger the date range change event manually for initial load
date_range_change_handler({"new": (default_start_date, default_end_date)})

semantic_score_slider = pn.widgets.FloatSlider(
    name="Similarity Percentage", start=0.1, end=1.0, step=0.01, value=0.95
)

semantic_score_slider = pn.widgets.FloatSlider(
    name="Similarity Percentage",
    start=0.1,
    end=1.0,
    step=0.01,
    value=0.95,
    css_classes=["material-slider"],  # Add the CSS class for styling
)

# Define clear section titles
response_search_title = pn.pane.Markdown(
    "## Response Similarity Search", css_classes=["widget-title"]
)
datetime_picker_title = pn.pane.Markdown(
    "### Select Date Range", css_classes=["widget-subtitle"]
)
quantity_selection_title = pn.pane.Markdown(css_classes=["widget-subtitle"])

response_layout_left = pn.Column(
    response_search_title,
    pn.Row(
        prompt_search_box,
        margin=(0, 20, 20, 0),  # Right margin to separate from slider
    ),
    search_button,
    pn.Row(
        semantic_score_slider,
        css_classes=["material-row"],
        margin=(0, 0, 0, 20),  # Left margin to separate from search controls
    ),
    pn.Row(
        "Response Similarity",
        prompt_quantity_select,
        css_classes=["material-row"],
        margin=(0, 0, 0, 20),  # Left margin to separate from search controls
    ),
    css_classes=["material-pane"],  # Apply consistent styling
    margin=(10, 10, 10, 10),
)

response_layout_right = pn.Column(
    pn.pane.Markdown("## Similar Responses", css_classes=["widget-title"]),
    pn.Row(response_similarity_table),
    pn.Row(retrieved_prompt_count_display, export_ef_controls),
    css_classes=["material-pane"],  # Apply consistent styling
    margin=(10, 10, 10, 10),
)


initialize_search_for_similar_prompts({"new": ("initalizing")})

# Setup the main layout
# Define a column for the search functionality and prompt display
prompt_history_layout_left = pn.Column(
    datetime_picker_title,
    datetime_range_picker,
    pn.Row("History", prompt_quantity_select),
    css_classes=["material-pane"],  # Apply consistent styling
    margin=(10, 10, 10, 10),
)

prompt_history_layout_right = pn.Column(
    pn.pane.Markdown("## Prompt:Response History", css_classes=["widget-title"]),
    pn.Row(prompt_history_table),
    pn.Row(retrieved_prompt_count_display, export_df_controls),
    # Group export controls and search functionality together for better coherence
    css_classes=["material-pane"],  # Apply consistent styling
    margin=(10, 10, 10, 10),
)

prompt_history_layout = pn.Column(
    prompt_history_layout_left,
    prompt_history_layout_right,
    css_classes=["material-pane"],
    # sizing_mode="stretch_width",  # Add this line to make the layout responsive
    margin=(10, 10, 10, 10),
)

response_similarity_layout = pn.Row(
    response_layout_left,
    response_layout_right,
    css_classes=["material-pane"],  # Apply consistent styling
    margin=(10, 10, 10, 10),
)


password_input = pn.widgets.PasswordInput(
    name="Password", placeholder="Enter your password here..."
)
username_input = pn.widgets.TextInput(
    name="Username", placeholder="Enter your username here..."
)
login_button = pn.widgets.Button(name="Login", button_type="primary")
login_button.on_click(login_to_promptmule)
login_display = pn.pane.Markdown(
    f"User: {promptmule_client.username}", css_classes=["material-label"]
)

logout_button = pn.widgets.Button(name="Logout", button_type="primary")
logout_button.on_click(logout_from_promptmule)


# Define a function to create a tab layout with consistent styling
def create_tab_layout(name, content):
    return (name, content)


response_tab = pn.Tabs(response_similarity_layout)
chat_tab = pn.Tabs(chat_interface)
history_tab = pn.Tabs(prompt_history_layout)
user_tab = pn.Tabs(username_input, password_input, login_button)

# Define the layout for the Response Similarity tab
response_similarity_tab_layout = pn.Row(
    response_layout_left,
    response_layout_right,
    css_classes=["material-pane"],  # Apply consistent styling
    margin=(10, 10, 10, 10),
)

username_input = pn.widgets.TextInput(
    name="Username", value="demo", placeholder="Enter username here"
)
integrations_input = pn.widgets.TextInput(
    name="Integrations", value="OpenAI", placeholder="Enter integrations here"
)
preference_slider = pn.widgets.FloatSlider(
    name="Preference Control", start=0.1, end=1.0, step=0.01, value=0.95
)


user_stats_output = promptmule_client.get_user_stats()
df_user_stats = pd.DataFrame(user_stats_output)
user_stats_display = pn.widgets.Tabulator(
    df_user_stats,
    pagination="local",
    page_size=10,
    sizing_mode="stretch_both",
    theme="materialize",
    layout="fit_columns",
    header_filters=True,
)

from bokeh.models import ColumnDataSource
from sklearn.cluster import KMeans


# Generate sample data
x = np.random.randint(1, 1000, size=100)
y = np.random.randint(1, 1000, size=100)

# Perform clustering
data = np.column_stack((x, y))
kmeans = KMeans(n_clusters=3, random_state=0).fit(data)
clusters = kmeans.labels_

# Create a color palette for clusters
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Add more colors if needed
cluster_colors = [colors[label] for label in clusters]

# Create a ColumnDataSource for plotting
source = ColumnDataSource(data=dict(x=x, y=y, color=cluster_colors))

# Create the scatter plot
p1 = figure(
    width_policy="max",
    height_policy="max",
    title="User Conversations: Mapping Prompt Diversity to Response Variety",
    sizing_mode="stretch_both",
    name="Use vs. Cache Hits",
    margin=5,
)

# Scatter plot with cluster coloring
p1.scatter("x", "y", source=source, color="color", size=8, alpha=0.6)

# Customize axes
p1.xaxis.axis_label = "Prompt Diversity"
p1.yaxis.axis_label = "Response Variety"
p1.xaxis.axis_label_text_font_size = "12pt"
p1.yaxis.axis_label_text_font_size = "12pt"

# Customize ticks
p1.xaxis.major_label_text_font_size = "12pt"
p1.yaxis.major_label_text_font_size = "12pt"
p1.xaxis.major_label_text_color = "navy"
p1.yaxis.major_label_text_color = "navy"

# Remove grid lines
p1.grid.grid_line_color = None

# Add a title
p1.title.text_font_size = "16pt"
p1.title.text_color = "black"

# Remove toolbar
p1.toolbar.logo = None
p1.toolbar_location = None


# Set the x_range to handle categorical data using the unique values from 'app-name'
p2 = figure(
    width_policy="max",
    height_policy="max",
    sizing_mode="stretch_both",
    title="Token Savings per App",
    x_range=FactorRange(*df_user_stats["app-name"]),
    margin=(5, 5, 5, 5),
)

# Customize X axis labels
p2.xaxis.major_label_orientation = 0.785  # Or use radians, e.g., 1.57 for horizontal
p2.xaxis.major_label_text_font_size = "12pt"  # Adjust font size
p2.xaxis.major_label_text_color = "navy"  # Adjust font color
p2.xaxis.axis_label_text_font_size = "12pt"  # Adjust axis label font size
p2.xaxis.axis_label_text_color = "navy"  # Adjust axis label color

# Customize Y axis labels
p2.yaxis.major_label_text_font_size = "12pt"  # Adjust font size
p2.yaxis.major_label_text_color = "navy"  # Adjust font color
p2.yaxis.axis_label_text_font_size = "12pt"  # Adjust axis label font size
p2.yaxis.axis_label_text_color = "navy"  # Adjust axis label color

# Customize the plot title
p2.title.text_font_size = "16pt"  # Adjust title font size
p2.title.text_color = "navy"  # Adjust title color

# Add a line renderer
p2.line(
    x=df_user_stats["app-name"],
    y=df_user_stats["saved-token"],
    line_width=5,
    color="orange",
)  # Adjust line color

profile = pn.Accordion(
    ("User", username_input),
    ("Preferences", preference_slider),
    ("Integrations", integrations_input),
    active=[],
)

# Define the layout for the History tab
history_tab_layout = pn.Row(
    prompt_history_layout_left,
    prompt_history_layout_right,
    css_classes=["material-pane"],  # Apply consistent styling
    margin=(10, 10, 10, 10),
)
user_tab_layout = pn.Column(
    "Login to PromptMule",
    pn.Row(username_input, password_input, login_button),
    login_display,
    logout_button,
    profile,
    css_classes=["material-pane"],  # Apply consistent styling
    margin=(10, 10, 10, 10),
)
# Create the History tab with consistent styling
response_similarity_tab = create_tab_layout(
    "Response Search", response_similarity_tab_layout
)
history_tab = create_tab_layout("Prompt History", history_tab_layout)
user_tab = create_tab_layout("User Profile", user_tab_layout)
analysis_tab = create_tab_layout("Prompt Analysis", p1)
savings_tab = create_tab_layout("Token Savings", p2)
stats_tab = create_tab_layout("Use Statistics", user_stats_display)

# Define the main layout with all tabs including the updated History tab
tabs = pn.Tabs(
    ("Chat", chat_interface),
    response_similarity_tab,
    history_tab,
    analysis_tab,
    savings_tab,
    stats_tab,
    user_tab,
    css_classes=["layout_background"],  # Apply background styling
    sizing_mode="stretch_width",
    margin=(10, 10, 10, 10),
)

# Define the main layout
main_layout = pn.Column(
    tabs,
    sizing_mode="stretch_width",  # Add this line to make the main layout responsive
    margin=(10, 10, 10, 10),
)

# Make the layout servable
main_layout.servable()

from langchain.tools import tool
from datetime import datetime
import os
import webbrowser
import urllib.parse


@tool
def get_datetime() -> str:
    """Returns the current date and time in human-readable format."""
    now = datetime.now()
    return now.strftime("It's %A, %B %d, %Y at %I:%M %p")

@tool
def delete_file_action(memory_file: str) -> str:
    """Delete a file"""
    # memory_file = "test.json"
    if os.path.exists(memory_file):
        os.remove(memory_file)
        return "File deleted successfully."
    else:
        return "File not found."
    
@tool
def youtube_search_opener(query: str) -> str:
    """
    Returns the YouTube search results URL for a given query.
    Actual browser opening is handled in the final step.
    """
    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://www.youtube.com/results?search_query={encoded_query}"
    return url
import streamlit as st
import google.generativeai as genai
import os
from pathlib import Path
import json
import time
import shutil  # For copying files

# --- UI Components ---
# These libraries provide pre-built UI elements like menus and the code editor.
from streamlit_option_menu import option_menu
from streamlit_ace import st_ace
import streamlit_antd_components as sac # Using for specific buttons (Save/Delete group)

# --- Configuration ---
st.set_page_config(
    layout="wide",
    page_title="AI Tool that generates another AI Tool" # Shorter title
)

# --- Constants ---
# Where generated Python app files will be saved
WORKSPACE_DIR = Path("workspace_st_apps")
WORKSPACE_DIR.mkdir(exist_ok=True) # Create the directory if it doesn't exist
PAGES_DIR = Path(".streamlit/pages")  # Streamlit's standard pages directory
PAGES_DIR.mkdir(parents=True, exist_ok=True)

# Code editor appearance settings
ACE_DEFAULT_THEME = "monokai"
ACE_DEFAULT_KEYBINDING = "vscode"

# Which Google AI model to use for generating code
GEMINI_MODEL_NAME = "gemini-2.5-pro-exp-03-25"

# Instructions for the Google AI model
# This tells the AI how to format its responses (as JSON commands)
GEMINI_SYSTEM_PROMPT = f"""
You are an AI assistant helping create Streamlit applications.
Your goal is to manage Python files in a workspace based on user requests.
Respond *only* with a valid JSON array containing commands. Do not add any explanations before or after the JSON array.

Available commands:
1.  `{{"action": "create_update", "filename": "app_name.py", "content": "FULL_PYTHON_CODE_HERE"}}`
    - Use this to create a new Python file or completely overwrite an existing one.
    - Provide the *entire* file content. Escape backslashes (`\\\\`) and double quotes (`\\"`). Ensure newlines are `\\n`.
    - Do *not* include ```python markdown blocks or shebangs (`#!/usr/bin/env python`) in the "content".
2.  `{{"action": "delete", "filename": "old_app.py"}}`
    - Use this to delete a Python file from the workspace.
3.  `{{"action": "chat", "content": "Your message here."}}`
    - Use this *only* if you need to ask for clarification, report an issue you can't fix with file actions, or confirm understanding.

Current Python files in workspace: {', '.join([f.name for f in WORKSPACE_DIR.iterdir() if f.is_file() and f.suffix == '.py']) if WORKSPACE_DIR.exists() else 'None'}

Example Interaction:
User: Create a simple hello world app called hello.py
AI: `[{{"action": "create_update", "filename": "hello.py", "content": "import streamlit as st\\n\\nst.title('Hello World!')\\nst.write('This is a simple app.')"}}`

Ensure your entire response is *only* the JSON array `[...]`.
"""

# --- API Client Setup ---
try:
    google_api_key = st.secrets["my_secrets"]["gemini_api_key"]
    if not google_api_key:
        # Stop the app if the API key is missing
        st.error("🔴 Google API Key not found. Please set `GOOGLE_API_KEY` in a `.env` file.")
        st.stop() # Halt execution
    # Configure the Gemini library with the key
    genai.configure(api_key=google_api_key)
    # Create the AI model object
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
except Exception as e:
    st.error(f"🔴 Failed to set up Google AI: {e}")
    st.stop()

# --- Session State ---
# Streamlit reruns the script on interaction. Session state stores data
# between reruns, like chat history or which file is selected.
def initialize_session_state():
    """Sets up default values in Streamlit's session state dictionary."""
    state_defaults = {
        "messages": [],             # List to store chat messages (user and AI)
        "selected_file": None,      # Name of the file currently shown in the editor
        "file_content_on_load": "", # Content of the selected file when loaded (read-only)
        "preview_process": None,    # Stores the running preview process object
        "preview_port": None,       # Port number used by the preview
        "preview_url": None,        # URL to access the preview
        "preview_file": None,       # Name of the file being previewed
        "editor_unsaved_content": "", # Current text typed into the editor
        "last_saved_content": "",   # Content that was last successfully saved to disk
    }
    for key, default_value in state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

initialize_session_state() # Run the initialization

# --- File System Functions ---
def get_workspace_python_files():
    """Gets a list of all '.py' filenames in the workspace directory."""
    if not WORKSPACE_DIR.is_dir():
        return [] # Return empty list if directory doesn't exist
    try:
        # List files, filter for .py, sort alphabetically
        python_files = sorted([
            f.name for f in WORKSPACE_DIR.iterdir() if f.is_file() and f.suffix == '.py'
        ])
        return python_files
    except Exception as e:
        st.error(f"Error reading workspace directory: {e}")
        return []

def read_file(filename):
    """Reads the text content of a file from the workspace."""
    if not filename: # Check if filename is provided
        return None
    # Prevent accessing files outside the workspace (basic security)
    if ".." in filename or filename.startswith(("/", "\\")):
        st.error(f"Invalid file path: {filename}")
        return None

    filepath = WORKSPACE_DIR / filename # Combine directory and filename
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read() # Return the file's text content
    except FileNotFoundError:
        st.warning(f"File not found: {filename}")
        return None # Indicate file doesn't exist
    except Exception as e:
        st.error(f"Error reading file '{filename}': {e}")
        return None

def save_file(filename, content):
    """Writes text content to a file in the workspace."""
    if not filename:
        return False # Cannot save without a filename
    if ".." in filename or filename.startswith(("/", "\\")):
        st.error(f"Invalid file path: {filename}")
        return False

    filepath = WORKSPACE_DIR / filename
    try:
        # Write the content to the file (overwrites if it exists)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return True # Indicate success
    except Exception as e:
        st.error(f"Error saving file '{filename}': {e}")
        return False # Indicate failure

def delete_file(filename):
    """Deletes a file from the workspace and updates app state."""
    if not filename:
        return False
    if ".." in filename or filename.startswith(("/", "\\")):
        st.error(f"Invalid file path: {filename}")
        return False

    filepath = WORKSPACE_DIR / filename
    try:
        if filepath.is_file():
            os.remove(filepath) # Delete the actual file
            # Also remove any preview of this file
            preview_path = PAGES_DIR / f"_preview_{filename}"
            if preview_path.exists():
                preview_path.unlink()
            
            st.toast(f"Deleted: {filename}", icon="🗑️")

            # If the deleted file was being previewed, stop the preview
            if st.session_state.preview_file == filename:
                stop_preview() # Call the function to stop the process

            # If the deleted file was selected in the editor, clear the selection
            if st.session_state.selected_file == filename:
                st.session_state.selected_file = None
                st.session_state.file_content_on_load = ""
                st.session_state.editor_unsaved_content = ""
                st.session_state.last_saved_content = ""
            return True # Indicate success
        else:
            st.warning(f"Could not delete: File '{filename}' not found.")
            return False
    except Exception as e:
        st.error(f"Error deleting file '{filename}': {e}")
        return False

# --- AI Interaction Functions ---

def _clean_ai_response_text(ai_response_text):
    """Removes potential code fences (```json ... ```) from AI response."""
    text = ai_response_text.strip()
    if text.startswith("```json"):
        text = text[7:-3].strip() # Remove ```json and ```
    elif text.startswith("```"):
        text = text[3:-3].strip() # Remove ``` and ```
    return text

def parse_and_execute_ai_commands(ai_response_text):
    """
    Parses the AI's JSON response and performs the requested file actions.
    Returns the list of commands (for chat history display).
    """
    cleaned_text = _clean_ai_response_text(ai_response_text)
    executed_commands_list = [] # To store commands for chat display

    try:
        # Attempt to convert the cleaned text into a Python list of dictionaries
        commands = json.loads(cleaned_text)

        # Check if the result is actually a list
        if not isinstance(commands, list):
            st.error("AI response was valid JSON, but not a list of commands.")
            # Return a chat message indicating the error for display
            return [{"action": "chat", "content": f"AI Error: Response was not a list. Response: {cleaned_text}"}]

        # Process each command dictionary in the list
        for command_data in commands:
            # Ensure the command is a dictionary before processing
            if not isinstance(command_data, dict):
                st.warning(f"AI sent an invalid command format (not a dict): {command_data}")
                executed_commands_list.append({"action": "chat", "content": f"AI Error: Invalid command format: {command_data}"})
                continue # Skip to the next command

            # Add the command to the list we return (used for displaying AI actions)
            executed_commands_list.append(command_data)

            # Get action details from the dictionary
            action = command_data.get("action")
            filename = command_data.get("filename")
            content = command_data.get("content")

            # --- Execute the action ---
            if action == "create_update":
                if filename and content is not None:
                    success = save_file(filename, content)
                    if success:
                        st.toast(f"AI saved: {filename}", icon="💾")
                        # If this file is currently open in the editor, update the editor's content
                        if st.session_state.selected_file == filename:
                            st.session_state.file_content_on_load = content
                            st.session_state.last_saved_content = content
                            st.session_state.editor_unsaved_content = content
                    else:
                        st.error(f"AI command failed: Could not save '{filename}'.")
                        # Add error details to chat display list
                        executed_commands_list.append({"action": "chat", "content": f"Error: Failed saving {filename}"})
                else:
                    st.warning("AI 'create_update' command missing filename or content.")
                    executed_commands_list.append({"action": "chat", "content": "AI Warning: Invalid create_update"})

            elif action == "delete":
                if filename:
                    success = delete_file(filename)
                    if not success:
                         st.error(f"AI command failed: Could not delete '{filename}'.")
                         executed_commands_list.append({"action": "chat", "content": f"Error: Failed deleting {filename}"})
                else:
                    st.warning("AI 'delete' command missing filename.")
                    executed_commands_list.append({"action": "chat", "content": "AI Warning: Invalid delete"})

            elif action == "chat":
                # No action needed here, the chat message is already in executed_commands_list
                # and will be displayed in the chat history.
                pass

            else:
                # Handle unrecognized actions from the AI
                st.warning(f"AI sent unknown action: '{action}'.")
                executed_commands_list.append({"action": "chat", "content": f"AI Warning: Unknown action '{action}'"})

        return executed_commands_list # Return the list for chat display

    except json.JSONDecodeError:
        st.error(f"AI response was not valid JSON.\nRaw response:\n```\n{cleaned_text}\n```")
        # Return a chat message indicating the JSON error for display
        return [{"action": "chat", "content": f"AI Error: Invalid JSON received. Response: {ai_response_text}"}]
    except Exception as e:
        st.error(f"Error processing AI commands: {e}")
        return [{"action": "chat", "content": f"Error processing commands: {e}"}]

def _prepare_gemini_history(chat_history, system_prompt):
    """Formats chat history for the Gemini API call."""
    gemini_history = []
    # Start with the system prompt (instructions for the AI)
    gemini_history.append({"role": "user", "parts": [{"text": system_prompt}]})
    # Gemini requires a model response to start the turn properly after a system prompt
    gemini_history.append({"role": "model", "parts": [{"text": json.dumps([{"action": "chat", "content": "Understood. I will respond only with JSON commands."}])}]})

    # Add the actual user/assistant messages from session state
    for msg in chat_history:
        role = msg["role"] # "user" or "assistant"
        content = msg["content"]
        api_role = "model" if role == "assistant" else "user" # Map to API roles

        # Convert assistant messages (which are lists of commands) back to JSON strings
        if role == "assistant" and isinstance(content, list):
            try:
                content_str = json.dumps(content)
            except Exception:
                content_str = str(content) # Fallback if conversion fails
        else:
            content_str = str(content) # User messages are already strings

        if content_str: # Avoid sending empty messages
            gemini_history.append({"role": api_role, "parts": [{"text": content_str}]})

    return gemini_history

def ask_gemini_ai(chat_history):
    """Sends the conversation history to the Gemini AI and returns its response."""

    # Get current list of files to include in the prompt context
    current_files = get_workspace_python_files()
    file_list_info = f"Current Python files: {', '.join(current_files) if current_files else 'None'}"
    # Update the system prompt with the current file list
    updated_system_prompt = GEMINI_SYSTEM_PROMPT.replace(
        "Current Python files: ...", # Placeholder text to replace
        file_list_info
    )

    # Prepare the history in the format the API expects
    gemini_api_history = _prepare_gemini_history(chat_history, updated_system_prompt)

    try:
        # Make the API call to Google
        # print(f"DEBUG: Sending history:\n{json.dumps(gemini_api_history, indent=2)}") # Uncomment for debugging API calls
        response = model.generate_content(gemini_api_history)
        # print(f"DEBUG: Received response:\n{response.text}") # Uncomment for debugging API calls
        return response.text # Return the AI's raw text response

    except Exception as e:
        # Handle potential errors during the API call
        error_message = f"Gemini API call failed: {type(e).__name__}"
        st.error(f"🔴 {error_message}: {e}")

        # Try to give a more user-friendly error message for common issues
        error_content = f"AI Error: {str(e)[:150]}..." # Default message
        if "API key not valid" in str(e):
            error_content = "AI Error: Invalid Google API Key."
        elif "429" in str(e) or "quota" in str(e).lower() or "resource has been exhausted" in str(e).lower():
            error_content = "AI Error: API Quota or Rate Limit Exceeded."
        # Handle cases where the AI's response might be blocked for safety
        try:
             if response and response.prompt_feedback and response.prompt_feedback.block_reason:
                 error_content = f"AI Error: Input blocked by safety filters ({response.prompt_feedback.block_reason})."
             elif response and response.candidates and response.candidates[0].finish_reason != 'STOP':
                  error_content = f"AI Error: Response stopped ({response.candidates[0].finish_reason}). May be due to safety filters or length limits."
        except Exception:
             pass # Ignore errors during safety check parsing

        # Return the error as a JSON chat command so it appears in the chat history
        return json.dumps([{"action": "chat", "content": error_content}])

# --- Live Preview Process Management ---
def stop_preview():
    """Stops the preview by removing the file from pages directory."""
    preview_file = st.session_state.get("preview_file")
    if preview_file:
        try:
            preview_path = PAGES_DIR / f"_preview_{preview_file}"
            if preview_path.exists():
                preview_path.unlink()
                st.toast(f"Preview stopped for {preview_file}", icon="⏹️")
        except Exception as e:
            st.error(f"Error stopping preview: {e}")
    
    # Clear preview state
    st.session_state.preview_file = None
    st.session_state.preview_url = None
    st.rerun()

def start_preview(python_filename):
    """Starts a preview by copying the file to the pages directory."""
    filepath = WORKSPACE_DIR / python_filename
    if not filepath.is_file() or filepath.suffix != '.py':
        st.error(f"Cannot preview: '{python_filename}' is not a valid Python file.")
        return False

    # Stop any existing preview
    stop_preview()

    try:
        # Create preview file in pages directory with a prefix to ensure ordering
        preview_path = PAGES_DIR / f"_preview_{python_filename}"
        
        # Copy the file content but add a title that shows it's a preview
        with open(filepath, 'r') as source:
            content = source.read()
        
        with open(preview_path, 'w') as dest:
            # Add a title at the top of the file
            preview_header = f'''import streamlit as st

# Preview of {python_filename}
st.title(f"🔍 Preview: {python_filename}")
st.divider()

# Original code below:
'''
            dest.write(preview_header + '\n' + content)

        # Update session state
        st.session_state.preview_file = python_filename
        st.session_state.preview_url = f"_preview_{python_filename}"  # Page URL will be lowercase filename
        st.toast(f"Preview started for {python_filename}", icon="🚀")
        return True

    except Exception as e:
        st.error(f"Error starting preview: {e}")
        return False

# --- Streamlit App UI ---

st.title("🤖 AI Tool that generates another AI Tool")

# --- Sidebar ---
with st.sidebar:
    st.header("💬 Chat & Controls")
    st.divider()

    # --- Chat History Display ---
    chat_container = st.container(height=400)
    with chat_container:
        if not st.session_state.messages:
            st.info("Chat history is empty. Type your instructions below.")
        else:
            # Loop through messages stored in session state
            for message in st.session_state.messages:
                role = message["role"] # "user" or "assistant"
                content = message["content"]
                avatar = "🧑‍💻" if role == "user" else "🤖"

                # Display message using Streamlit's chat message element
                with st.chat_message(role, avatar=avatar):
                    if role == "assistant" and isinstance(content, list):
                        # Assistant message contains commands - format them nicely
                        file_actions_summary = ""
                        chat_responses = []
                        code_snippets = []

                        for command in content:
                            if not isinstance(command, dict): continue # Skip malformed

                            action = command.get("action")
                            filename = command.get("filename")
                            cmd_content = command.get("content")

                            if action == "create_update":
                                file_actions_summary += f"📝 **Saved:** `{filename}`\n"
                                if cmd_content:
                                    code_snippets.append({"filename": filename, "content": cmd_content})
                            elif action == "delete":
                                file_actions_summary += f"🗑️ **Deleted:** `{filename}`\n"
                            elif action == "chat":
                                chat_responses.append(str(cmd_content or "..."))
                            else:
                                file_actions_summary += f"⚠️ **Unknown Action:** `{action}`\n"

                        # Display the formatted summary and chat responses
                        full_display_text = (file_actions_summary + "\n".join(chat_responses)).strip()
                        if full_display_text:
                            st.markdown(full_display_text)
                        else: # Handle cases where AI might return empty actions
                             st.markdown("(AI performed no displayable actions)")

                        # Show code snippets in collapsible sections
                        for snippet in code_snippets:
                            with st.expander(f"View Code for `{snippet['filename']}`", expanded=False):
                                st.code(snippet['content'], language="python")

                    elif isinstance(content, str):
                        # Simple text message (from user or AI chat action)
                        st.write(content)
                    else:
                        # Fallback for unexpected content type
                        st.write(f"Unexpected message format: {content}")

    # --- Chat Input Box ---
    user_prompt = st.chat_input("Tell the AI what to do (e.g., 'Create hello.py')")
    if user_prompt:
        # 1. Add user's message to the chat history (in session state)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        # 2. Show a spinner while waiting for the AI
        with st.spinner("🧠 AI Thinking..."):
            # 3. Send the *entire* chat history to the AI
            ai_response_text = ask_gemini_ai(st.session_state.messages)
            # 4. Parse the AI's response and execute file commands
            ai_commands_executed = parse_and_execute_ai_commands(ai_response_text)

        # 5. Add the AI's response (the list of executed commands) to chat history
        st.session_state.messages.append({"role": "assistant", "content": ai_commands_executed})

        # 6. Rerun the script immediately to show the new messages and update file list/editor
        st.rerun()

    st.divider()

    # --- Status Info ---
    st.subheader("Status & Info")
    st.success(f"Using AI model: {GEMINI_MODEL_NAME}", icon="✅")
    st.warning(
        "**Notes:** Review AI code before running previews. `create_update` overwrites files.",
    )


# --- Main Area Tabs ---
selected_tab = option_menu(
    menu_title=None,
    options=["Workspace", "Live Preview"],
    icons=["folder-fill", "play-btn-fill"],
    orientation="horizontal",
    key="main_tab_menu"
    # Removed custom styles for simplicity
)

# --- Workspace Tab ---
if selected_tab == "Workspace":
    st.header("📂 Workspace & Editor")
    st.divider()

    # Create two columns: one for file list, one for editor
    file_list_col, editor_col = st.columns([0.3, 0.7]) # 30% width for files, 70% for editor

    with file_list_col:
        st.subheader("Files")
        python_files = get_workspace_python_files()

        # Prepare options for the dropdown menu
        select_options = ["--- Select a file ---"] + python_files
        current_selection_in_state = st.session_state.get("selected_file")

        # Find the index of the currently selected file to set the dropdown default
        try:
            current_index = select_options.index(current_selection_in_state) if current_selection_in_state else 0
        except ValueError:
            current_index = 0 # If file in state doesn't exist, default to "Select"

        # The dropdown widget
        selected_option = st.selectbox(
            "Edit file:",
            options=select_options,
            index=current_index,
            key="file_selector_dropdown",
            label_visibility="collapsed" # Hide the label "Edit file:"
        )

        # --- Handle File Selection Change ---
        # If the dropdown selection is different from what's stored in session state...
        newly_selected_filename = selected_option if selected_option != "--- Select a file ---" else None
        if newly_selected_filename != current_selection_in_state:
            st.session_state.selected_file = newly_selected_filename # Update state
            # Read the content of the newly selected file
            file_content = read_file(newly_selected_filename) if newly_selected_filename else ""
            # Handle case where file read failed (e.g., it was deleted)
            if file_content is None and newly_selected_filename:
                 file_content = f"# ERROR: Could not read file '{newly_selected_filename}'"

            # Update session state with the file's content for the editor
            st.session_state.file_content_on_load = file_content
            st.session_state.editor_unsaved_content = file_content # Start editor with file content
            st.session_state.last_saved_content = file_content     # Mark as saved initially
            st.rerun() # Rerun script to load the new file into the editor

    with editor_col:
        st.subheader("Code Editor")
        selected_filename = st.session_state.selected_file

        if selected_filename:
            st.caption(f"Editing: `{selected_filename}`")

            # Display the Ace code editor widget
            editor_current_text = st_ace(
                value=st.session_state.get('editor_unsaved_content', ''), # Show unsaved content
                language="python",
                theme=ACE_DEFAULT_THEME,
                keybinding=ACE_DEFAULT_KEYBINDING,
                font_size=14, tab_size=4, wrap=True,
                auto_update=False, # Don't trigger reruns on every keystroke
                key=f"ace_editor_{selected_filename}" # Unique key helps reset state on file change
            )

            # Check if the editor's current text is different from the last saved text
            has_unsaved_changes = (editor_current_text != st.session_state.last_saved_content)

            # If the text in the editor box changes, update our 'unsaved' state variable
            if editor_current_text != st.session_state.editor_unsaved_content:
                st.session_state.editor_unsaved_content = editor_current_text
                st.rerun() # Rerun to update the 'Save Changes' button state

            # --- Editor Action Buttons ---
            # Using sac.buttons here for the nice grouped layout with icons.
            editor_buttons = [
                sac.ButtonsItem(label="💾 Save Changes", icon="save", disabled=not has_unsaved_changes),
                sac.ButtonsItem(label="🗑️ Delete File", icon="trash", color="red"),
            ]
            clicked_editor_button = sac.buttons(
                 items=editor_buttons, index=None, format_func='title',
                 align='end', size='small', return_index=False,
                 key="editor_action_buttons"
            )

            # --- Handle Button Clicks ---
            if clicked_editor_button == "💾 Save Changes":
                if save_file(selected_filename, editor_current_text):
                    # Update state to reflect the save
                    st.session_state.file_content_on_load = editor_current_text
                    st.session_state.last_saved_content = editor_current_text
                    st.toast(f"Saved: `{selected_filename}`", icon="💾")
                    time.sleep(0.5) # Let toast message show
                    st.rerun() # Rerun to disable the save button
                else:
                    st.error(f"Error: Could not save '{selected_filename}'.")

            elif clicked_editor_button == "🗑️ Delete File":
                 # Use sac.confirm_button for a confirmation pop-up
                 needs_confirmation = True # Flag to show confirmation
                 if needs_confirmation:
                      confirmed = sac.confirm_button(
                          f"Delete `{selected_filename}`?", # Confirmation message
                          color="error", key="confirm_delete_button"
                      )
                      if confirmed:
                          if delete_file(selected_filename):
                              # Deletion successful, file list and editor will update on rerun
                              st.rerun()
                          # No 'else' needed, delete_file shows errors

            # Show a warning if there are unsaved changes
            if has_unsaved_changes:
                st.warning("You have unsaved changes.")

        else:
            # Show a placeholder message if no file is selected
            st.info("Select a Python file from the list on the left to view or edit.")
            st_ace(value="# Select a file...", language="python", readonly=True, key="ace_placeholder")

# --- Live Preview Tab ---
elif selected_tab == "Live Preview":
    st.header("▶️ Live Preview")
    st.divider()
    st.warning("⚠️ Running AI-generated code can have unintended consequences. Review code first!")

    # Get preview status from session state
    file_being_previewed = st.session_state.get("preview_file")
    preview_url = st.session_state.get("preview_url")
    selected_file_for_preview = st.session_state.get("selected_file") # File selected in Workspace

    # --- Preview Controls ---
    st.subheader("Controls")
    if not selected_file_for_preview:
        st.info("Select a file in the 'Workspace' tab to enable preview controls.")
        # Allow stopping a preview even if no file is selected
        if file_being_previewed:
            st.warning(f"Preview is running for: `{file_being_previewed}`")
            if st.button(f"⏹️ Stop Preview ({file_being_previewed})", key="stop_other_preview"):
                stop_preview()
    else:
        # Controls for the file selected in the Workspace
        st.write(f"File selected for preview: `{selected_file_for_preview}`")
        is_python = selected_file_for_preview.endswith(".py")

        if not is_python:
            st.error("Cannot preview: Selected file is not a Python (.py) file.")
        else:
            # Layout Run and Stop buttons side-by-side
            run_col, stop_col = st.columns(2)
            with run_col:
                # Disable Run button if this file is already being previewed
                run_disabled = (file_being_previewed == selected_file_for_preview)
                if st.button("🚀 Run Preview", disabled=run_disabled, type="primary", use_container_width=True):
                    if start_preview(selected_file_for_preview):
                        st.rerun()
            with stop_col:
                # Disable Stop button if no preview is running
                stop_disabled = not file_being_previewed
                if st.button("⏹️ Stop Preview", disabled=stop_disabled, use_container_width=True):
                    stop_preview()

    st.divider()

    # --- Preview Display ---
    st.subheader("Preview Window")
    if file_being_previewed:
        # Check if the running preview matches the file selected in the workspace
        if file_being_previewed == selected_file_for_preview:
            st.success(f"Preview is running! Click the page '🔍 Preview: {file_being_previewed}' in the sidebar to view it.")
            st.info("Tip: You can keep both the main app and preview open in separate browser tabs.")
        else:
            # A preview is running, but not for the file selected in the workspace
            st.warning(f"Preview is running for `{file_being_previewed}`. Select that file in the Workspace to see it here, or stop it using the controls above.")
    else:
        # No preview is currently running
        st.info("Click 'Run Preview' on a selected Python file to see it here.")
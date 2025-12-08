"""
The Brain - NBA Betting Co-Pilot
Streamlit Chat Interface

Run with: streamlit run chat/app.py
"""
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env file
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

import anthropic

from chat.prompts import SYSTEM_PROMPT
from chat.tools import TOOLS, execute_tool


# Page config
st.set_page_config(
    page_title="The Brain",
    page_icon="🧠",
    layout="centered"
)

# Custom CSS for cleaner chat
st.markdown("""
<style>
    .stChatMessage {
        padding: 0.5rem 1rem;
    }
    .stChatMessage p {
        margin-bottom: 0.5rem;
    }
    /* Make responses more scannable */
    .stChatMessage ul, .stChatMessage ol {
        margin: 0.25rem 0;
        padding-left: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session():
    """Initialize session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_id" not in st.session_state:
        # Demo user - in production this would come from auth
        st.session_state.user_id = "demo-user-001"


def get_client():
    """Get Anthropic client."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("ANTHROPIC_API_KEY not set. Please set it in your environment.")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


def chat(user_message: str, history: list) -> tuple[str, list]:
    """
    Send message and get response, handling tool calls.

    Args:
        user_message: User's message
        history: Conversation history

    Returns:
        (response_text, updated_history)
    """
    client = get_client()

    # Build messages
    messages = history + [{"role": "user", "content": user_message}]

    # Initial API call
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=messages
    )

    # Handle tool use loop
    while response.stop_reason == "tool_use":
        tool_calls = [block for block in response.content if block.type == "tool_use"]

        tool_results = []
        for tool_call in tool_calls:
            result = execute_tool(
                tool_name=tool_call.name,
                tool_input=tool_call.input,
                user_id=st.session_state.user_id
            )
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": json.dumps(result)
            })

        # Add assistant response and tool results
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        # Continue conversation
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages
        )

    # Extract text response
    text_response = ""
    for block in response.content:
        if hasattr(block, "text"):
            text_response += block.text

    # Update history
    messages.append({"role": "assistant", "content": response.content})

    return text_response, messages


def main():
    """Main app."""
    init_session()

    # Header
    st.title("The Brain")
    st.caption("NBA Betting Co-Pilot")

    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                # Handle both string and list content
                content = msg["content"]
                if isinstance(content, str):
                    st.write(content)
                elif isinstance(content, list):
                    for block in content:
                        if hasattr(block, "text"):
                            st.write(block.text)

    # Quick action buttons (only show if no messages)
    if not st.session_state.messages:
        st.write("**Try asking:**")
        cols = st.columns(3)

        prompts = [
            "Best props tonight?",
            "4-leg parlay for any game",
            "What's your sleeper tonight?"
        ]

        for i, prompt in enumerate(prompts):
            if cols[i].button(prompt, key=f"quick_{i}"):
                st.session_state.quick_prompt = prompt
                st.rerun()

    # Handle quick prompt
    if hasattr(st.session_state, "quick_prompt") and st.session_state.quick_prompt:
        user_input = st.session_state.quick_prompt
        st.session_state.quick_prompt = None

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, updated_history = chat(user_input, [])
                st.write(response)

        # Save to session
        st.session_state.messages = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response}
        ]

    # Chat input
    if user_input := st.chat_input("Ask about props, parlays, or tonight's games..."):
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Build history from stored messages
                history = []
                for msg in st.session_state.messages:
                    if msg["role"] in ["user", "assistant"]:
                        history.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })

                response, updated_history = chat(user_input, history)
                st.write(response)

        # Save to session
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()

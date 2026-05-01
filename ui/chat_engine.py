"""
Chat UI Engine
Renders the conversational interface and handles Quick Action Pills
to maintain a seamless, form-free UX.

Changes from original:
- Pills use st.pills() (available in Streamlit 1.40+) instead of st.button()
  in columns, so labels never word-wrap and multi-row layout is handled natively.
"""

import streamlit as st


def render_chat():
    """Renders existing chat history and interactive pills."""
    for i, msg in enumerate(st.session_state.messages):
        # Strict role enforcement prevents "unknown ()" avatar glitch
        role = msg.get("role", "assistant")

        with st.chat_message(role):
            st.markdown(msg.get("content", ""))

            # Render pills inline below the assistant message that requested them
            if role == "assistant" and msg.get("render_pills"):
                pill_data = msg["render_pills"]
                for field_key, options in pill_data.items():
                    # st.pills() renders compact non-wrapping pill buttons natively.
                    # selection_mode="single" means only one option can be chosen.
                    # key includes message index so pills for different messages
                    # don't share state.
                    selected = st.pills(
                        label=field_key,           # hidden label (screen-reader only)
                        options=options,
                        selection_mode="single",
                        key=f"pills_{i}_{field_key}",
                        label_visibility="collapsed",
                    )
                    if selected:
                        st.session_state.pill_clicked = {
                            "field": field_key,
                            "value": selected,
                        }
                        st.rerun()


def process_pill_if_clicked():
    """Checks if a pill was clicked and converts it into a natural user message."""
    if "pill_clicked" in st.session_state and st.session_state.pill_clicked:
        pill      = st.session_state.pill_clicked
        user_text = f"I've selected {pill['value']}."
        st.session_state.messages.append({"role": "user", "content": user_text})
        del st.session_state.pill_clicked
        return True
    return False

import streamlit as st
from transformers import pipeline

# Page Configuration
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 AI Text Summarizer")
st.markdown("Generate concise summaries from long-form text using AI.")

# Load Model
@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )

with st.spinner("Loading AI Model..."):
    summarizer = load_model()

# Text Input
text = st.text_area(
    "Paste your text here",
    height=300,
    placeholder="Enter article, blog, research paper, or any long text..."
)

# Statistics
col1, col2 = st.columns(2)

with col1:
    st.metric("Characters", len(text))

with col2:
    st.metric("Words", len(text.split()))

# Summary Length
summary_length = st.selectbox(
    "Choose Summary Length",
    ["Short", "Medium", "Long"]
)

# Length Settings
if summary_length == "Short":
    max_len = 50
    min_len = 20

elif summary_length == "Medium":
    max_len = 100
    min_len = 40

else:
    max_len = 150
    min_len = 60

# Buttons
col1, col2 = st.columns(2)

generate = col1.button("Generate Summary")
clear = col2.button("Clear Text")

# Clear Functionality
if clear:
    st.rerun()

# Generate Summary
if generate:

    if not text.strip():

        st.warning("⚠ Please enter some text.")

    else:

        with st.spinner("Generating Summary..."):

            try:

                result = summarizer(
                    text,
                    max_length=max_len,
                    min_length=min_len,
                    do_sample=False,
                    truncation=True
                )

                summary = result[0]["summary_text"]

                st.subheader("📄 Summary")

                st.success(summary)

                st.subheader("📋 Copy Summary")

                st.code(summary)

                st.download_button(
                    label="⬇ Download Summary",
                    data=summary,
                    file_name="summary.txt",
                    mime="text/plain"
                )

            except Exception as e:

                st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.markdown(
    "Built using Python, Streamlit, Hugging Face Transformers, and NLP."
)
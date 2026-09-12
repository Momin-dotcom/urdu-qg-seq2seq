import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "model"))

from decode import generate

st.title("Urdu Question Generation")

sentence = st.text_area("Paste an Urdu sentence:")
answer = st.text_input("Mark the answer (paste the exact answer span from the sentence):")

if st.button("Generate Question"):
    if not sentence or not answer:
        st.warning("Please provide both the sentence and the answer.")
    elif answer not in sentence:
        st.warning("The answer text must appear exactly as written in the sentence.")
    else:
        with st.spinner("Generating..."):
            greedy_output = generate(sentence, answer, method="greedy")
            beam_output = generate(sentence, answer, method="beam")

        st.subheader("Greedy Decoding")
        st.write(greedy_output)

        st.subheader("Beam Search Decoding")
        st.write(beam_output)
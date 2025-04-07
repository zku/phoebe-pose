import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from typing import Generator
from glob import glob
import streamlit.web.bootstrap
import streamlit as st
import traceback

load_dotenv()

ROOT_DIRECTORY = Path(os.getcwd())

TITLE = "Phoebe Image Generator"

INITIAL_PROMPT = """
# Introduction

You are an AI image understanding and generation expert.
Provided are multiple images of a border terrier dog called "Phoebe", the sweetest little elderly girl.
Your task is to analyze these images and extract the essence of this dog (i.e. her face, coloring, demeanor, posture, character, etc.)
and then create a new image of Phoebe in a different environment.

# Images of Phoebe
""".strip()

TASK_PROMPT = """
# Task

Follow these guidelines unless specifically instructed otherwise:

- Maintain her general look and demeanor
- Maintain her colors and proportions
- Maintain the shape and features of her face - this is very important

Using the reference images of Phoebe, please generate a new image of Phoebe with the following properties:

{user_instructions}
""".strip()

EXAMPLE_PROMPTS = [
    "Phoebe sitting in a wheat field like in Gladiator. Ghibli comic style.",
    "Professional photo shoot of Phoebe, close up. Fotorealistic. Black background. Smooth lighting.",
]


def load_images() -> Generator[bytes, None, None]:
    """Loads reference images from disk."""
    assets_dir = ROOT_DIRECTORY / "assets"
    for image_path in glob("**/*.png", root_dir=assets_dir):
        with open(assets_dir / image_path, "rb") as f:
            yield f.read()


def generate_images(user_instructions: str) -> Generator[bytes, None, None]:
    """Generates one or more images from the given user instructions."""

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    contents: types.Content = []
    contents.append(types.Content(role="user", parts=[types.Part(text=INITIAL_PROMPT)]))
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(inline_data=types.Blob(data=data, mime_type="image/png")) for data in load_images()],
        )
    )
    contents.append(
        types.Content(role="user", parts=[types.Part(text=TASK_PROMPT.format(user_instructions=user_instructions))])
    )

    generation_config = types.GenerateContentConfig(
        temperature=1.0,
        response_modalities=[types.Modality.IMAGE, types.Modality.TEXT],
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash-exp-image-generation", contents=contents, config=generation_config
    )

    for candidate in response.candidates:
        for part in candidate.content.parts:
            if part.inline_data:
                yield part.inline_data.data


def main_streamlit() -> None:
    """Streamlit app entry point. This is executed every frame."""
    st.set_page_config(page_title=TITLE)
    st.title(TITLE)

    for prompt in EXAMPLE_PROMPTS:
        st.code(prompt, language="markdown")

    prompt = st.text_input("Enter image description")
    if prompt:
        with st.spinner("Generating image...", show_time=True):
            for image_bytes in generate_images(prompt):
                st.image(image_bytes)


for frame in traceback.extract_stack():
    if "streamlit" in frame.filename:
        main_streamlit()
        break


def main() -> None:
    streamlit.web.bootstrap.run(__file__, False, [], {})

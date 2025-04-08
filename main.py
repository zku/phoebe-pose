"""Generate AI images of Phoebe."""

import asyncio
import concurrent
import os
from glob import glob
from pathlib import Path
from typing import Generator

import streamlit as st
from google import genai
from google.genai import types

ROOT_DIRECTORY = Path(os.getcwd())

NUM_IMAGE_GENERATIONS = 3
TITLE = "Phoebe Image Generator"

INITIAL_PROMPT = """
# Introduction

You are an AI image understanding and generation expert.
Provided are multiple images of a border terrier dog called "Phoebe",
the sweetest little elderly girl. Your task is to analyze these images
and extract the essence of this dog (i.e. her face, coloring, demeanor,
posture, character, etc.) and then create a new image of Phoebe in a
different environment.

# Images of Phoebe
""".strip()

TASK_PROMPT = """
# Task

Follow these guidelines unless specifically instructed otherwise:

- Maintain her general look and demeanor
- Maintain her colors and proportions
- Maintain the shape and features of her face - this is very important

Using the reference images of Phoebe, please generate a new image of Phoebe
with the following properties:

{user_instructions}
""".strip()

EXAMPLE_PROMPTS = [
    "Phoebe sitting in a wheat field like in Gladiator. Ghibli comic style.",
    "Cyberpunk illustration of Phoebe in dimly lit streets of an Asian city.",
]


def load_images() -> Generator[bytes, None, None]:
    """Load reference images from disk."""
    assets_dir = ROOT_DIRECTORY / "assets"
    for image_path in glob("**/*.png", root_dir=assets_dir):
        with open(assets_dir / image_path, "rb") as f:
            yield f.read()


def generate_images(user_instructions: str) -> list[bytes]:
    """Generate one or more images from the given user instructions."""
    parts = [
        types.Part(text=INITIAL_PROMPT),
        *[
            types.Part(inline_data=types.Blob(data=data, mime_type="image/png"))
            for data in load_images()
        ],
        types.Part(text=TASK_PROMPT.format(user_instructions=user_instructions)),
    ]
    generation_config = types.GenerateContentConfig(
        temperature=1.0,
        # We only care about images, but the API requires us to accept TEXT too.
        response_modalities=[types.Modality.IMAGE, types.Modality.TEXT],
    )

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp-image-generation",
        contents=types.Content(role="user", parts=parts),
        config=generation_config,
    )

    return [
        part.inline_data.data
        for candidate in response.candidates
        for part in candidate.content.parts
        if part.inline_data and part.inline_data.data
    ]


async def generate_images_async(prompt: str, count: int = 1) -> list[bytes]:
    """Generate multiple images for the same prompt."""
    loop = asyncio.get_running_loop()
    tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as executor:
        for _ in range(count):
            task_future = loop.run_in_executor(executor, generate_images, prompt)
            tasks.append(task_future)
    results = await asyncio.gather(*tasks)
    return [image for images in results for image in images]


def main_streamlit() -> None:
    """Streamlit app entry point. This is executed every frame."""
    st.set_page_config(page_title=TITLE, layout="centered")
    st.header("🐕 " + TITLE, divider="rainbow")

    for prompt in EXAMPLE_PROMPTS:
        st.code(prompt, language="markdown")
    if prompt := st.text_area("Enter image description"):
        with st.spinner("Generating image(s)...", show_time=True):
            all_images = asyncio.run(generate_images_async(prompt, NUM_IMAGE_GENERATIONS))
        for col, image_bytes in zip(st.columns(NUM_IMAGE_GENERATIONS), all_images):
            col.image(image_bytes)

if __name__ == "__main__":
    main_streamlit()

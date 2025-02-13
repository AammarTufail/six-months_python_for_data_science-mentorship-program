from lida import Manager, TextGenerationConfig , llm  
from dotenv import load_dotenv
import os
import openai
import base64
from PIL import Image
from io import BytesIO

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

def base64_to_image(base64_string):
    # Decode the base64 string
    byte_data = base64.b64decode(base64_string)
    
    # Use BytesIO to convert the byte data to image
    return Image.open(BytesIO(byte_data))

def save_image(base64_str, save_path):
    img = base64_to_image(base64_str)
    img.save(save_path)
    print(f"Image saved at {save_path}")

lida = Manager(text_gen = llm("openai")) 
textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
summary = lida.summarize("2019.csv", summary_method="default", textgen_config=textgen_config)

user_query = "Which country has the most GDP per capita?"
charts = lida.visualize(summary=summary, goal=user_query, textgen_config=textgen_config)  
charts[0]

image_base64 = charts[0].raster

save_image(image_base64, "filename1.png")


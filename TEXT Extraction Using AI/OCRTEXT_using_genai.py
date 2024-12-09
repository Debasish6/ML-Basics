import sys
import re
import json
from pathlib import Path
import google.generativeai as genai
import mimetypes

# Configure the Generative AI model
genai.configure(api_key='AIzaSyBJDEKz0SeHcq98hLj3L_yh6fFryNWyO8k')

MODEL_CONFIG = {
    "temperature": 0.2,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

# Safety Settings of Model
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
]

model = genai.GenerativeModel(model_name="gemini-1.5-flash",
                              generation_config=MODEL_CONFIG,
                              safety_settings=safety_settings)

def file_format(file_path):
    try:
      file = Path(file_path)

      if not file.exists():
          raise FileNotFoundError(f"Could not find file: {file}")

      mime_type, _ = mimetypes.guess_type(file_path)
      if mime_type not in ["image/png", "image/jpeg", "image/webp", "application/pdf"]:
          raise ValueError(f"Unsupported file type: {mime_type}")

      file_parts = [
          {
              "mime_type": mime_type,
              "data": file.read_bytes()
          }
      ]
      return file_parts
    except FileNotFoundError as fnf_error:
        print(f"Error: {fnf_error}")
    except Exception as e:
        print(f"Unexpected error occurred: {e}") 



# def image_format(image_path):
#     try:
#         img = Path(image_path)

#         if not img.exists():
#             raise FileNotFoundError(f"Could not find image: {img}")

#         image_parts = [{"mime_type": "image/png", "data": img.read_bytes()}]  # MIME type can be adjusted based on file format
#         return image_parts

#     except FileNotFoundError as fnf_error:
#         print(f"Error: {fnf_error}")
#     except Exception as e:
#         print(f"Unexpected error occurred: {e}")

def extract_text(image_path, system_prompt, user_prompt):
    try:
        image_info = file_format(image_path)
        if not image_info:
            raise ValueError("Invalid image information")

        input_prompt = [system_prompt, image_info[0], user_prompt]
        response = model.generate_content(input_prompt)
        return response.text

    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")

def remove_markdown_syntax(markdown_json):
    try:
        pattern = r'```json\n|\n```'
        cleaned_json = re.sub(pattern, '', markdown_json).strip()
        json_object = json.loads(cleaned_json)

        json_object = json.dumps(json_object, indent=4)

        with open(r"C:\Users\eDominer\Downloads\extracted_text.json", "w") as outfile:
            outfile.write(json_object)

        return json_object

    except json.JSONDecodeError as json_error:
        print(f"JSON decode error: {json_error}")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")

if __name__ == '__main__':
    try:
        image_path = r"C:\Users\eDominer\Downloads\FileServer (3).pdf"
        system_prompt = """
        You are a specialist in comprehending receipts.
        Input images in the form of receipts will be provided to you,
        and your task is to respond to questions based on the content of the input image.
        """
        user_prompt = "Convert Invoice data into json format with appropriate json tags as required for the data in image"

        output = extract_text(image_path, system_prompt, user_prompt)
        if output:
            cleaned_json_object = remove_markdown_syntax(output)
            print(cleaned_json_object)
        else:
            print("Failed to extract text from image")

    except Exception as e:
        print(f"An unexpected error occurred in the main function: {e}")

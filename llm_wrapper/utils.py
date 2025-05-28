import math
import re
import base64
from PIL import Image
import io
import requests

def check_input_structure(input_data):
    """
    Checks if the input_data is structured correctly.

    Args:
        input_data (dict): The input data to check.

    Returns:
        tuple: (bool, str) where the first value indicates if the input is valid,
               and the second value is an error message if the input is invalid.
    """
    if not isinstance(input_data, dict):
        return False, "Input is not a dictionary."

    for key, value in input_data.items():
        if not isinstance(value, dict):
            return False, f"Value for key '{key}' is not a dictionary."
        
        if "user_prompt" not in value:
            return False, f"Key 'user_prompt' missing in dictionary for key '{key}'."
        
        if not isinstance(value["user_prompt"], str):
            return False, f"Value for 'user_prompt' in key '{key}' is not a string."

        for sub_key, sub_value in value.items():
            if sub_key == "system_prompt":
                if not isinstance(sub_value, str):
                    return False, f"Value for 'system_prompt' in key '{key}' is not a string."
            elif sub_key == "image_url":
                if not isinstance(sub_value, str):
                    return False, f"Value for 'image_url' in key '{key}' is not a string."
            elif sub_key == "seed":
                if not isinstance(sub_value, int):
                    return False, f"Value for 'seed' in key '{key}' is not an integer."
            elif sub_key != "user_prompt":
                return False, f"Invalid key '{sub_key}' found in dictionary for key '{key}'."
    
    return True, "Input is valid."

def get_image_dimensions_from_url(image_url):
    # Get the image from the URL
    response = requests.get(image_url, stream=True)
    
    if response.status_code == 200:
        response.raw.decode_content = True
        image = Image.open(response.raw)
        return image.size  # (width, height)
    else:
        return None, None

def get_image_dimensions_from_base64(encoded_image):
    image_data = base64.b64decode(encoded_image)
    image = Image.open(io.BytesIO(image_data))
    return image.width, image.height

def extract_encoding(image_input):
    pattern = r'data:image/jpeg;base64,([^"]+)'

    match = re.search(pattern, image_input)
    if match:
        base64_part = match.group(1)
    else:
        base64_part = None
    return base64_part

def is_url(string):
    pattern = re.compile(
        r'^(https?|ftp):\/\/'          # protocol
        r'(([A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+'  # domain name
        r'[A-Z]{2,6}\.?|'              # domain extension
        r'localhost|'                  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # IPv4 address
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)' # IPv6 address
        r'(?::\d+)?'                   # optional port
        r'(\/?|[\/?]\S+)$',            # resource path
        re.IGNORECASE
    )
    return re.match(pattern, string) is not None


def tokens_from_image_input(image_input, detail='high'):

    if not is_url(image_input):
        image_encoding = extract_encoding(image_input)
        if image_encoding is None:
            return 1000
        
        width, height = get_image_dimensions_from_base64(image_encoding)
    else:
        width, height = get_image_dimensions_from_url(image_input)
        if width is None:
            return 1000
        
    if detail == 'low':
        return 85

    # Scale down to fit within a 2048 x 2048 square if necessary
    if width > 2048 or height > 2048:
        max_size = 2048
        aspect_ratio = width / height
        if aspect_ratio > 1:
            width = max_size
            height = int(max_size / aspect_ratio)
        else:
            height = max_size
            width = int(max_size * aspect_ratio)

    # Resize such that the shortest side is 768px if the original dimensions exceed 768px
    min_size = 768
    aspect_ratio = width / height
    if width > min_size and height > min_size:
        if aspect_ratio > 1:
            height = min_size
            width = int(min_size * aspect_ratio)
        else:
            width = min_size
            height = int(min_size / aspect_ratio)

    tiles_width = math.ceil(width / 512)
    tiles_height = math.ceil(height / 512)
    return 85 + 170 * (tiles_width * tiles_height)
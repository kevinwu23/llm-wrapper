# Standard library imports
import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime

# Third-party imports
import anthropic
import google.generativeai as genai
import pandas as pd
from openai import OpenAI

# Local imports
try:
    # Try relative imports first (when used as package)
    from .api_request_parallel_processor import process_api_requests_from_file
    from .utils import check_input_structure
except ImportError:
    # Fall back to absolute imports (when used directly)
    from api_request_parallel_processor import process_api_requests_from_file
    from utils import check_input_structure

# Automatically load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional - continue without it
    pass

logging.basicConfig(level=logging.CRITICAL)  # Suppress all logging below CRITICAL level

from tqdm import tqdm
from pathlib import Path
from typing import List, Optional

def generate_hash(input_string, length=5):
    """Generate a truncated SHA-256 hash of the input string."""
    hash_object = hashlib.sha256(input_string.encode())
    hash_hex = hash_object.hexdigest()
    return hash_hex[:length]

def extract_all_tags(text):
    """Extract all XML-style tags and their content from text."""
    if not isinstance(text, str):
        return {}
    pattern = r'<(\w+)>(.*?)</\1>'
    matches = re.findall(pattern, text, re.DOTALL)
    return {tag: content.strip() for tag, content in matches}

def encode_image(image_path):
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class BaseWrapper(ABC):
    def __init__(self, 
                 model_name=None,
                 response_format=None,
                 system_prompt=None,
                 seed=None,
                 temperature=0,
                 logprobs=None,
                 batch_path='temp'):
        
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.response_format = response_format
        self.seed = seed
        self.temperature = temperature
        self.logprobs = logprobs
        self.batch_path = batch_path
        self.f_out = None
        
        # Default rate limits (can be overridden by child classes)
        self.max_requests_per_minute = 1000
        self.max_tokens_per_minute = 100000

    @abstractmethod
    def __call__(self, user_prompt, system_prompt=None, **kwargs):
        """Make a single API call. Must be implemented by child classes."""
        pass

    @abstractmethod
    def _get_request_url(self):
        """Get the API request URL. Must be implemented by child classes."""
        pass

    @abstractmethod
    def _get_api_key(self):
        """Get the API key. Must be implemented by child classes."""
        pass

    @abstractmethod
    def _create_request_payload(self, query_key, prompt_dict):
        """Create the request payload for a single query. Must be implemented by child classes."""
        pass

    @abstractmethod
    def _extract_response_content(self, result, output_type="response"):
        """Extract content from API response. Must be implemented by child classes."""
        pass

    def check_batch(self, f_out=None):
        """Check the progress of a batch job."""
        if f_out is None:
            if self.f_out is None:
                print('No f_out specified')
                return
            f_out = self.f_out
            print(f'Using most recent run at {f_out}')
        
        f_in = f_out.replace('output.jsonl', 'input.jsonl')
        completed = sum(1 for _ in open(f_out)) if os.path.exists(f_out) else 0
        total = sum(1 for _ in open(f_in)) if os.path.exists(f_in) else 0
        print(f"{completed}/{total} completed")

    def run_batch(self, query_dict, run_background=False):
        """Run a batch of queries asynchronously."""
        if self.batch_path is None:
            raise ValueError("You need to set batch_path before running a batch job.")

        check, reason = check_input_structure(query_dict)
        if not check:
            raise ValueError(f"The given query dictionary format is invalid: {reason}")

        # Setup input/output files
        f_hash = generate_hash(datetime.now().strftime('%d/%m/%Y %H:%M:%S'))
        f_in = f"temp/{f_hash}_input.jsonl"
        f_out = f"temp/{f_hash}_output.jsonl"
        total_queries = len(query_dict)
        
        # Create input file
        self._write_input_file(f_in, query_dict)

        # Start async processing
        thread = self._start_async_processing(f_in, f_out)

        if run_background:
            self.f_out = f_out
            return f_out

        # Monitor progress
        self._monitor_progress(thread, f_out, total_queries)
        
        self.f_out = f_out
        return f_out

    def _write_input_file(self, f_in, query_dict):
        """Write the input JSONL file for batch processing."""
        with open(f_in, 'w') as f:
            for query_key, prompt_dict in query_dict.items():
                job_line = self._create_request_payload(query_key, prompt_dict)
                json_string = json.dumps(job_line)
                f.write(json_string + "\n")

    def _start_async_processing(self, f_in, f_out):
        """Start the async API processing in a separate thread."""
        def run_async_process():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(process_api_requests_from_file(
                requests_filepath=f_in,
                save_filepath=f_out,
                request_url=self._get_request_url(),
                api_key=self._get_api_key(),
                max_requests_per_minute=self.max_requests_per_minute,
                max_tokens_per_minute=self.max_tokens_per_minute,
                token_encoding_name='cl100k_base',
                max_attempts=5,
                logging_level=logging.CRITICAL
            ))
            loop.close()

        thread = threading.Thread(target=run_async_process)
        thread.start()
        return thread

    def _monitor_progress(self, thread, f_out, total_queries):
        """Monitor and display progress of batch processing."""
        completed = 0
        while thread.is_alive():
            if os.path.exists(f_out):
                current_completed = sum(1 for _ in open(f_out))
                if current_completed > completed:
                    print(f"\rProcessing batch: {current_completed}/{total_queries} ({(current_completed/total_queries)*100:.1f}%)", end="")
                    completed = current_completed
            time.sleep(0.1)

        # Final progress update
        if os.path.exists(f_out):
            final_completed = sum(1 for _ in open(f_out))
            print(f"\rProcessing batch: {final_completed}/{total_queries} ({(final_completed/total_queries)*100:.1f}%)")

        thread.join()

    def process_batch(self, f_out=None, output_type="response"):
        """Process batch results and return a dictionary of responses."""
        if f_out is None:
            f_out = self.f_out
            print(f'Using most recent run at {f_out}')
            
        output_response = {}
        with open(f_out, 'r') as f:
            for line in f:
                try:
                    result = json.loads(line)
                    query_key = result[-1]  # Metadata is always the last element
                    
                    # Convert unhashable metadata keys to string
                    try:
                        hash(query_key)
                    except TypeError:
                        query_key = json.dumps(query_key, sort_keys=True)
                    
                    completion = self._extract_response_content(result, output_type)
                    output_response[query_key] = completion
                    
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue

        return output_response

    def results_df(self, f_out=None, column_names=None):
        """Get batch results as a pandas DataFrame with extracted tags."""
        # Get batch results using existing process_batch method
        batch_results = self.process_batch(f_out)
        
        # Create a DataFrame
        df = pd.DataFrame.from_dict(batch_results, orient='index')
        
        # Rename the original column
        df = df.rename(columns={0: 'raw_response'})
        
        # If column_names is -1, return the DataFrame with just the raw_response column
        if column_names == -1:
            try:
                # Try to convert index to integer, but handle string indices gracefully
                df.index = pd.to_numeric(df.index, errors='ignore')
                return df.sort_index()
            except:
                # If sorting fails, return unsorted
                return df
        
        # Extract all tags and their content
        tag_data = df['raw_response'].apply(extract_all_tags)
        
        if column_names is None:
            # Get all unique tags across all rows
            all_tags = set()
            for tags in tag_data:
                all_tags.update(tags.keys())
            
            # Create new columns for each tag
            for tag in all_tags:
                df[tag] = tag_data.apply(lambda x: x.get(tag, None))
        else:
            # Create new columns only for specified tags
            for tag in column_names:
                df[tag] = tag_data.apply(lambda x: x.get(tag, None))
        
        # Convert index to integer if possible, otherwise keep as is
        try:
            df.index = pd.to_numeric(df.index, errors='ignore')
            # Sort by index if possible
            df = df.sort_index()
        except:
            # If conversion or sorting fails, keep as is
            pass

        # If column_names is specified, keep only those columns
        if column_names is not None and column_names != -1:
            df = df[column_names]
        else:
            # Only remove raw_response if it's not the only column
            if len(df.columns) > 1 and 'raw_response' in df.columns:
                del df['raw_response']
        
        return df

class OpenAIWrapper(BaseWrapper):
    def __init__(self, model_name=None, 
                 response_format=None, 
                 system_prompt=None, 
                 seed=42, 
                 temperature=0, 
                 logprobs=None,
                 batch_path=None,
                 api_key=None):

        super().__init__(model_name, response_format, system_prompt, seed, temperature, logprobs, batch_path)
        
        # Get API key from parameter or environment variable
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
        
        if api_key is None:
            raise ValueError(
                "OpenAI API key is required. Either:\n"
                "1. Set the OPENAI_API_KEY environment variable, or\n"
                "2. Pass api_key parameter: OpenAIWrapper(api_key='your-key')"
            )
        
        self.client = OpenAI(api_key=api_key)
        
        if self.model_name is None:
            self.model_name = "gpt-4o"
        
        if response_format == 'json':
            self.response_format = {"type": 'json_object'}

        # Override rate limits
        self.max_requests_per_minute = 10000
        self.max_tokens_per_minute = 2000000

    def _get_request_url(self):
        return 'https://api.openai.com/v1/chat/completions'

    def _get_api_key(self):
        return self.client.api_key

    def _create_request_payload(self, query_key, prompt_dict):
        messages = []
        if 'system_prompt' in prompt_dict:
            messages.append({"role": "system", "content": prompt_dict['system_prompt']})
        elif self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})

        if 'image_url' in prompt_dict:
            if 'http' not in prompt_dict['image_url']:
                base64_image = encode_image(prompt_dict['image_url'])
                image_url = f"data:image/jpeg;base64,{base64_image}"
            else:
                image_url = prompt_dict['image_url']
            messages.append({"role": "user", "content": [
                {"type": "text", "text": prompt_dict['user_prompt']},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]})
        else:
            messages.append({"role": "user", "content": prompt_dict['user_prompt']})
        
        return {
            "model": self.model_name,
            "messages": messages,
            "response_format": self.response_format,
            "seed": prompt_dict.get('seed', self.seed),
            "metadata": str(query_key)
        }

    def _extract_response_content(self, result, output_type="response"):
        if output_type == "response":
            try:
                return result[1]['choices'][0]['message']['content']
            except:
                return None
        else:  # logprobs
            try:
                return str(tuple(x['logprob'] for x in result[1]['choices'][0]['logprobs']['content']))
            except:
                return None

    def __call__(self, user_prompt, system_prompt=None, image_urls=None):
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        elif self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        
        if image_urls is not None:
            message_contents = [{"type": "text", "text": user_prompt}]
            for image_url in image_urls:
                if 'http' not in image_url:
                    base64_image = encode_image(image_url)
                    image_url = f"data:image/jpeg;base64,{base64_image}"
                message_contents.append({"type": "image_url", "image_url": {"url": image_url}})
            messages.append({"role": "user", "content": message_contents})
        else:
            messages.append({"role": "user", "content": user_prompt})
            
        completion = self.client.chat.completions.create(
          model=self.model_name,
          response_format=self.response_format,
          seed=self.seed,
          messages=messages
        )
        return completion.choices[0].message.content

class GeminiWrapper(BaseWrapper):
    def __init__(self, api_key=None, **kwargs):
        super().__init__(**kwargs)
        
        # Get API key from parameter or environment variable
        if api_key is None:
            api_key = os.getenv('GEMINI_API_KEY')
        
        if api_key is None:
            raise ValueError(
                "Gemini API key is required. Either:\n"
                "1. Set the GEMINI_API_KEY environment variable, or\n"
                "2. Pass api_key parameter: GeminiWrapper(api_key='your-key')"
            )
        
        self.gemini_api = api_key
        genai.configure(api_key=self.gemini_api)
        
        if self.model_name is None:
            self.model_name = 'gemini-2.5-flash'
            
        self.safety_settings = [
            {
              "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
              "threshold": "BLOCK_NONE"
            },
            {
              "category": "HARM_CATEGORY_HATE_SPEECH",
              "threshold": "BLOCK_NONE"
            },
            {
              "category": "HARM_CATEGORY_HARASSMENT",
              "threshold": "BLOCK_NONE"
            },
            {
              "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
              "threshold": "BLOCK_NONE"
            }
        ]
        
        # Override rate limits
        self.max_requests_per_minute = 100000
        self.max_tokens_per_minute = 10000000

    def _get_request_url(self):
        return f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.gemini_api}"

    def _get_api_key(self):
        return self.gemini_api

    def _create_request_payload(self, query_key, prompt_dict):
        contents = [{"role": "user", "parts": [{"text": prompt_dict['user_prompt']}]}]
        
        return {
            "model": self.model_name,
            "contents": contents,
            "safety_settings": self.safety_settings,
            "metadata": str(query_key)
        }

    def _extract_response_content(self, result, output_type="response"):
        try:
            return result[1]['candidates'][0]['content']['parts'][0]['text']
        except:
            return None

    def __call__(self, user_prompt, system_prompt=None):
        model = genai.GenerativeModel(self.model_name, system_instruction=system_prompt)
        response = model.generate_content(user_prompt)
    
        try:
            if len(response.candidates[0].content.parts) == 0:
                return "N/A"
            response = response.candidates[0].content.parts[0].text
        except:
            return "N/A"
        return response

class ClaudeWrapper(BaseWrapper):
    def __init__(self, api_key=None, **kwargs):
        super().__init__(**kwargs)
        
        # Get API key from parameter or environment variable
        if api_key is None:
            api_key = os.getenv('CLAUDE_API_KEY')
        
        if api_key is None:
            raise ValueError(
                "Claude API key is required. Either:\n"
                "1. Set the CLAUDE_API_KEY environment variable, or\n"
                "2. Pass api_key parameter: ClaudeWrapper(api_key='your-key')"
            )
        
        self.client = anthropic.Anthropic(api_key=api_key)
        
        if self.model_name is None:
            self.model_name = "claude-3-5-sonnet-20240620"
            
        # Override rate limits
        self.max_requests_per_minute = 2000
        self.max_tokens_per_minute = 80000

    def _get_request_url(self):
        return "https://api.anthropic.com/v1/messages"

    def _get_api_key(self):
        return self.client.api_key

    def _create_request_payload(self, query_key, prompt_dict):
        messages = [{"role": "user", "content": prompt_dict['user_prompt']}]
        
        if 'system_prompt' in prompt_dict:
            system_prompt = prompt_dict['system_prompt']
        elif self.system_prompt is not None:
            system_prompt = self.system_prompt
        else:
            system_prompt = None

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": self.temperature,
            "metadata": str(query_key)
        }
        
        if system_prompt is not None:
            payload["system"] = system_prompt
            
        return payload

    def _extract_response_content(self, result, output_type="response"):
        if output_type == "response":
            try:
                return result[1]['content'][0]['text']
            except:
                return None
        elif output_type == "logprobs":
            try:
                return str(tuple([x['logprob'] for x in result[1]['choices'][0]['logprobs']['content']]))
            except:
                return None

    def __call__(self, user_prompt, system_prompt=None, image_urls=None):
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        elif self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        
        messages.append({"role": "user", "content": user_prompt})
            
        completion = self.client.messages.create(
          model=self.model_name,
          max_tokens=1024,
          messages=messages
        )
        return completion.content[0].text

class TogetherAIWrapper(BaseWrapper):
    def __init__(self, api_key=None, **kwargs):
        super().__init__(**kwargs)
        
        # Get API key from parameter or environment variable
        if api_key is None:
            api_key = os.getenv('TOGETHER_API_KEY')
        
        if api_key is None:
            raise ValueError(
                "Together AI API key is required. Either:\n"
                "1. Set the TOGETHER_API_KEY environment variable, or\n"
                "2. Pass api_key parameter: TogetherAIWrapper(api_key='your-key')"
            )
        
        self.client = OpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=api_key
        )
        
        if self.model_name is None:
            self.model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
            
        # Updated rate limits for Together AI
        self.max_requests_per_minute = 4500
        self.max_tokens_per_minute = 1000000

    def _get_request_url(self):
        return "https://api.together.xyz/v1/chat/completions"

    def _get_api_key(self):
        return self.client.api_key

    def _create_request_payload(self, query_key, prompt_dict):
        messages = []
        if 'system_prompt' in prompt_dict:
            messages.append({"role": "system", "content": prompt_dict['system_prompt']})
        elif self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})

        messages.append({"role": "user", "content": prompt_dict['user_prompt']})
        
        return {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "metadata": str(query_key)
        }

    def _extract_response_content(self, result, output_type="response"):
        try:
            # Check if this is a Llama-3.1 model response
            if isinstance(result, list) and len(result) == 3 and "Llama-3.1" in str(result[0].get("model", "")):
                query_key = result[2]  # The key is the third element
                if output_type == "response":
                    return result[1]['choices'][0]['message']['content']
                elif output_type == "logprobs":
                    return str(tuple([x['logprob'] for x in result[1]['choices'][0]['logprobs']['content']]))
            else:
                # Handle standard response format
                if output_type == "response":
                    return result[1]['choices'][0]['message']['content']
                elif output_type == "logprobs":
                    return str(tuple([x['logprob'] for x in result[1]['choices'][0]['logprobs']['content']]))
        except:
            return None

    def __call__(self, user_prompt, system_prompt=None, image_urls=None):
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        elif self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        
        messages.append({"role": "user", "content": user_prompt})
            
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature
        )
        return completion.choices[0].message.content
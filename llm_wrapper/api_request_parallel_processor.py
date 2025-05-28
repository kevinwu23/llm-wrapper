"""
API REQUEST PARALLEL PROCESSOR

Using the OpenAI API to process lots of text quickly takes some care.
If you trickle in a million API requests one by one, they'll take days to complete.
If you flood a million API requests in parallel, they'll exceed the rate limits and fail with errors.
To maximize throughput, parallel requests need to be throttled to stay under rate limits.

This script parallelizes requests to the OpenAI API while throttling to stay under rate limits.

Features:
- Streams requests from file, to avoid running out of memory for giant jobs
- Makes requests concurrently, to maximize throughput
- Throttles request and token usage, to stay under rate limits
- Retries failed requests up to {max_attempts} times, to avoid missing data
- Logs errors, to diagnose problems with requests

Example command to call script:
```
python examples/api_request_parallel_processor.py \
  --requests_filepath examples/data/example_requests_to_parallel_process.jsonl \
  --save_filepath examples/data/example_requests_to_parallel_process_results.jsonl \
  --request_url https://api.openai.com/v1/embeddings \
  --max_requests_per_minute 1500 \
  --max_tokens_per_minute 6250000 \
  --token_encoding_name cl100k_base \
  --max_attempts 5 \
  --logging_level 20
```

Inputs:
- requests_filepath : str
    - path to the file containing the requests to be processed
    - file should be a jsonl file, where each line is a json object with API parameters and an optional metadata field
    - e.g., {"model": "text-embedding-ada-002", "input": "embed me", "metadata": {"row_id": 1}}
    - as with all jsonl files, take care that newlines in the content are properly escaped (json.dumps does this automatically)
    - an example file is provided at examples/data/example_requests_to_parallel_process.jsonl
    - the code to generate the example file is appended to the bottom of this script
- save_filepath : str, optional
    - path to the file where the results will be saved
    - file will be a jsonl file, where each line is an array with the original request plus the API response
    - e.g., [{"model": "text-embedding-ada-002", "input": "embed me"}, {...}]
    - if omitted, results will be saved to {requests_filename}_results.jsonl
- request_url : str, optional
    - URL of the API endpoint to call
    - if omitted, will default to "https://api.openai.com/v1/embeddings"
- api_key : str, optional
    - API key to use
    - if omitted, the script will attempt to read it from an environment variable {os.getenv("OPENAI_API_KEY")}
- max_requests_per_minute : float, optional
    - target number of requests to make per minute (will make less if limited by tokens)
    - leave headroom by setting this to 50% or 75% of your limit
    - if requests are limiting you, try batching multiple embeddings or completions into one request
    - if omitted, will default to 1,500
- max_tokens_per_minute : float, optional
    - target number of tokens to use per minute (will use less if limited by requests)
    - leave headroom by setting this to 50% or 75% of your limit
    - if omitted, will default to 125,000
- token_encoding_name : str, optional
    - name of the token encoding used, as defined in the `tiktoken` package
    - if omitted, will default to "cl100k_base" (used by `text-embedding-ada-002`)
- max_attempts : int, optional
    - number of times to retry a failed request before giving up
    - if omitted, will default to 5
- logging_level : int, optional
    - level of logging to use; higher numbers will log fewer messages
    - 40 = ERROR; will log only when requests fail after all retries
    - 30 = WARNING; will log when requests his rate limits or other errors
    - 20 = INFO; will log when requests start and the status at finish
    - 10 = DEBUG; will log various things as the loop runs to see when they occur
    - if omitted, will default to 20 (INFO).

The script is structured as follows:
    - Imports
    - Define main()
        - Initialize things
        - In main loop:
            - Get next request if one is not already waiting for capacity
            - Update available token & request capacity
            - If enough capacity available, call API
            - The loop pauses if a rate limit error is hit
            - The loop breaks when no tasks remain
    - Define dataclasses
        - StatusTracker (stores script metadata counters; only one instance is created)
        - APIRequest (stores API inputs, outputs, metadata; one method to call API)
    - Define functions
        - api_endpoint_from_url (extracts API endpoint from request URL)
        - append_to_jsonl (writes to results file)
        - num_tokens_consumed_from_request (bigger function to infer token usage from request)
        - task_id_generator_function (yields 0, 1, 2, ...)
    - Run main()
"""

# imports
import aiohttp  # for making API calls concurrently
import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import re  # for matching endpoint from request URL
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
from dataclasses import (
    dataclass,
    field,
)  # for storing API inputs, outputs, and metadata
try:
    # Try relative import first (when used as package)
    from .utils import tokens_from_image_input
except ImportError:
    # Fall back to absolute import (when used directly)
    from utils import tokens_from_image_input
from asyncio import Semaphore

async def process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    request_url: str,
    api_key: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name: str,
    max_attempts: int = 5,
    logging_level: int = logging.INFO,
    request_header: dict = None,
) -> None:
    """Process API requests in parallel from a file."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = (
        .001  # 1 ms limits max throughput to 1,000 requests per second
    )
    max_attempts = 100

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # infer API endpoint and construct request header
    api_endpoint = api_endpoint_from_url(request_url)
    request_header = {"Authorization": f"Bearer {api_key}"}
    # use api-key header for Azure deployments
    # if '/deployments' in request_url:
    #     request_header = {"api-key": f"{api_key}"}
    if 'anthropic' in request_url:
        request_header = {"x-api-key": f"{api_key}", "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
    elif 'deepinfra' in request_url:
        request_header = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    elif 'aiplatform.googleapis.com' in request_url:
        request_header = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    elif 'google' in request_url:
        request_header = {"Content-Type": "application/json"}
    elif 'stanfordhealthcare' in request_url:
        request_header = {"Ocp-Apim-Subscription-Key": api_key, "Content-Type": "application/json"}
    else:
        request_header = {"Authorization": f"Bearer {api_key}"}
    
    # Check if using o3-mini model and adjust rate limits accordingly
    if isinstance(requests_filepath, str) and os.path.exists(requests_filepath):
        with open(requests_filepath) as f:
            try:
                first_request = json.loads(f.readline())
                if isinstance(first_request, dict):
                    model = first_request.get("model", "")
                    if first_request.get("model") == "o3-mini":
                        max_requests_per_minute = 10000
                        max_tokens_per_minute = 10000000
                        logging.info(f"Detected o3-mini model, setting rate limits to {max_requests_per_minute} RPM and {max_tokens_per_minute} TPM")
                    elif "R1" in str(model):
                        max_requests_per_minute = 300
                        max_tokens_per_minute = 300000
                        logging.info(f"Detected Together AI R1 model, setting rate limits to {max_requests_per_minute} RPM and {max_tokens_per_minute} TPM")
                    elif "3-7" in str(model) and "claude" in str(model).lower():
                        max_tokens_per_minute = 200000
                        logging.info(f"Detected Claude 3-7 model, setting token rate limit to {max_tokens_per_minute} TPM")
                    elif "together.xyz" in request_url:  # Add check for Together AI API
                        max_requests_per_minute = 4500
                        max_tokens_per_minute = 1000000
                        logging.info(f"Detected Together AI API, setting rate limits to {max_requests_per_minute} RPM and {max_tokens_per_minute} TPM")
            except (json.JSONDecodeError, StopIteration):
                pass
    
    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = (
        task_id_generator_function()
    )  # generates integer IDs of 0, 1, 2, ...
    status_tracker = (
        StatusTracker()
    )  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug(f"Initialization complete.")

    # initialize file reading
    with open(requests_filepath) as file:
        # `requests` will provide requests one at a time
        requests = file.__iter__()
        logging.debug(f"File opened. Entering main loop")
        async with aiohttp.ClientSession() as session:  # Initialize ClientSession here
            semaphore = Semaphore(200)
            while True:
                # get next request (if one is not already waiting for capacity)
                if next_request is None:
                    if not queue_of_requests_to_retry.empty():
                        next_request = queue_of_requests_to_retry.get_nowait()
                        logging.debug(
                            f"Retrying request {next_request.task_id}: {next_request}"
                        )
                    elif file_not_finished:
                        try:
                            # get new request
                            request_json = json.loads(next(requests))

                            next_request = APIRequest(
                                task_id=next(task_id_generator),
                                request_json=request_json,
                                token_consumption=num_tokens_consumed_from_request(
                                    request_json, api_endpoint, token_encoding_name
                                ),
                                attempts_left=max_attempts,
                                metadata=request_json.pop("metadata", None),
                            )
                        
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                            logging.debug(
                                f"Reading request {next_request.task_id}: {next_request}"
                            )
                        except StopIteration:
                            # if file runs out, set flag to stop reading it
                            logging.debug("Read file exhausted")
                            file_not_finished = False

                # update available capacity
                current_time = time.time()
                seconds_since_update = current_time - last_update_time
                available_request_capacity = min(
                    available_request_capacity
                    + max_requests_per_minute * seconds_since_update / 60.0,
                    max_requests_per_minute,
                )
                available_token_capacity = min(
                    available_token_capacity
                    + max_tokens_per_minute * seconds_since_update / 60.0,
                    max_tokens_per_minute,
                )
                last_update_time = current_time

                # if enough capacity available, call API
                if next_request:
                    next_request_tokens = next_request.token_consumption
                    if (
                        available_request_capacity >= 1
                        and available_token_capacity >= next_request_tokens
                    ):
                        # update counters
                        available_request_capacity -= 1
                        available_token_capacity -= next_request_tokens
                        next_request.attempts_left -= 1

                        # call API

                        asyncio.create_task(
                            next_request.call_api(
                                session=session,
                                request_url=request_url,
                                request_header=request_header,
                                retry_queue=queue_of_requests_to_retry,
                                save_filepath=save_filepath,
                                status_tracker=status_tracker,
                            )
                        )

                        next_request = None  # reset next_request to empty

                # if all tasks are finished, break
                if status_tracker.num_tasks_in_progress == 0:
                    break

                # main loop sleeps briefly so concurrent tasks can run
                await asyncio.sleep(seconds_to_sleep_each_loop)

                # if a rate limit error was hit recently, pause to cool down
                seconds_since_rate_limit_error = (
                    time.time() - status_tracker.time_of_last_rate_limit_error
                )
                if (
                    seconds_since_rate_limit_error
                    < seconds_to_pause_after_rate_limit_error
                ):
                    remaining_seconds_to_pause = (
                        seconds_to_pause_after_rate_limit_error
                        - seconds_since_rate_limit_error
                    )
                    await asyncio.sleep(remaining_seconds_to_pause)
                    # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                    logging.warn(
                        f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
                    )

        # after finishing, log final status
        logging.info(
            f"""Parallel processing complete. Results saved to {save_filepath}"""
        )
        if status_tracker.num_tasks_failed > 0:
            logging.warning(
                f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}."
            )
        if status_tracker.num_rate_limit_errors > 0:
            logging.warning(
                f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
            )


# dataclasses


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ):
        """Calls the API and saves results."""
        logging.info(f"Starting request #{self.task_id}")
        error = None
        try:
            async with session.post(
                url=request_url, 
                headers=request_header, 
                json=self.request_json
            ) as response:
                response = await response.json()
            
            # Check for Google AI Platform rate limit error
            if isinstance(response, dict) and "error" in response:
                error_details = response["error"]
                if (
                    error_details.get("code") == 429 
                    or "RESOURCE_EXHAUSTED" in error_details.get("status", "")
                    or "Rate limit" in error_details.get("message", "")
                ):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    # Force retry by raising an exception
                    raise Exception("Rate limit exceeded")
                else:
                    logging.warning(f"Request {self.task_id} failed with error {response['error']}")
                    status_tracker.num_api_errors += 1
                    error = response

            # Check for list response with error (your specific case)
            elif isinstance(response, list) and len(response) > 0 and isinstance(response[0], dict):
                if "error" in response[0]:
                    error_details = response[0]["error"]
                    if (
                        error_details.get("code") == 429 
                        or "RESOURCE_EXHAUSTED" in error_details.get("status", "")
                        or "Rate limit" in error_details.get("message", "")
                    ):
                        status_tracker.time_of_last_rate_limit_error = time.time()
                        status_tracker.num_rate_limit_errors += 1
                        # Force retry by raising an exception
                        raise Exception("Rate limit exceeded")
                    else:
                        logging.warning(f"Request {self.task_id} failed with error {response[0]['error']}")
                        status_tracker.num_api_errors += 1
                        error = response

        except (
            Exception
        ) as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(
                    f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}"
                )
                data = (
                    [self.request_json, [str(e) for e in self.result], self.metadata]
                    if self.metadata
                    else [self.request_json, [str(e) for e in self.result]]
                )
                append_to_jsonl(data, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = (
                [self.request_json, response, self.metadata]
                if self.metadata
                else [self.request_json, response]
            )
            append_to_jsonl(data, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} saved to {save_filepath}")


# functions


def api_endpoint_from_url(request_url):
    """Extract the API endpoint from the request URL."""

    if 1:
        return request_url
    match = re.search("^https://[^/]+/v\\d+/(.+)$", request_url)
    if match is None:
        # for Azure OpenAI deployment urls
        match = re.search(r"^https://[^/]+/openai/deployments/[^/]+/(.+?)(\?|$)", request_url)
    return match[1]


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")

def find_tag(data, tag):
    urls = []
    if isinstance(data, dict):
        for k, v in data.items():
            if k == tag:
                urls.append(v)
            elif isinstance(v, dict):
                urls.extend(find_tag(v, tag))
            elif isinstance(v, list):
                for item in v:
                    urls.extend(find_tag(item, tag))
    elif isinstance(data, list):
        for item in data:
            urls.extend(find_tag(item, tag))
    return urls


def num_tokens_consumed_from_request(request_json: dict, *args, **kwargs) -> int:
    """Count the number of tokens in the request. Handles both OpenAI and Gemini formats."""
    
    # Handle Gemini format from the file
    if isinstance(request_json, list) and len(request_json) == 3:
        try:
            # Extract the prompt from the contents
            request_data = request_json[0]
            if "contents" in request_data:
                for content in request_data["contents"]:
                    if content["role"] == "user":
                        for part in content["parts"]:
                            if "text" in part:
                                # Approximate token count (4 chars ~= 1 token)
                                return len(part["text"]) // 4
        except (KeyError, IndexError):
            pass
    
    # Handle OpenAI format
    elif isinstance(request_json, dict) and "messages" in request_json:
        num_tokens = 0
        for message in request_json["messages"]:
            num_tokens += len(message["content"]) // 4
        return num_tokens
    
    # If neither format is found
    return 0


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1


# run script


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests_filepath")
    parser.add_argument("--save_filepath", default=None)
    parser.add_argument("--request_url", default="https://api.openai.com/v1/embeddings")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--max_requests_per_minute", type=int, default=3_000 * 0.5)
    parser.add_argument("--max_tokens_per_minute", type=int, default=250_000 * 0.5)
    parser.add_argument("--token_encoding_name", default="cl100k_base")
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument("--logging_level", default=logging.INFO)
    args = parser.parse_args()

    if args.save_filepath is None:
        args.save_filepath = args.requests_filepath.replace(".jsonl", "_results.jsonl")

    # run script
    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=args.requests_filepath,
            save_filepath=args.save_filepath,
            request_url=args.request_url,
            api_key=args.api_key,
            max_requests_per_minute=float(args.max_requests_per_minute),
            max_tokens_per_minute=float(args.max_tokens_per_minute),
            token_encoding_name=args.token_encoding_name,
            max_attempts=int(args.max_attempts),
            logging_level=int(args.logging_level),
        )
    )


"""
APPENDIX

The example requests file at openai-cookbook/examples/data/example_requests_to_parallel_process.jsonl contains 10,000 requests to text-embedding-ada-002.

It was generated with the following code:

```python
import json

filename = "data/example_requests_to_parallel_process.jsonl"
n_requests = 10_000
jobs = [{"model": "text-embedding-ada-002", "input": str(x) + "\n"} for x in range(n_requests)]
with open(filename, "w") as f:
    for job in jobs:
        json_string = json.dumps(job)
        f.write(json_string + "\n")
```

As with all jsonl files, take care that newlines in the content are properly escaped (json.dumps does this automatically).
"""

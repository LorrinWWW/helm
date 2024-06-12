import os
from copy import deepcopy
from typing import List, Dict, Any, Optional, Union

import requests
from retrying import retry

from helm.common.cache import CacheConfig
from helm.common.request import wrap_request_time, Request, RequestResult, Sequence, Token
from .client import CachingClient, truncate_sequence, cleanup_str

def post_processing_text(output_text, stop_tokens, denylist = []):

    filtered_stop_tokens = []
    for token in stop_tokens:
        if token != '':
            filtered_stop_tokens.append(token)

    end_pos = len(output_text)
    for stop_token in filtered_stop_tokens:
        if output_text.find(stop_token) != -1:
            end_pos = min(output_text.find(stop_token), end_pos)

    post_processed_text = output_text[:end_pos]
    for word in denylist:
        if post_processed_text.find(word) != -1:
            print(f"<post_processing_text> post_processed_text: {post_processed_text}")
            print(f"<post_processing_text> denylist word {word} found, set to empty.")
            post_processed_text = "Sorry, I'm not sure how to answer that question."
            break
    return post_processed_text


class PulsarClientError(Exception):
    pass


class JobNotFinishedError(PulsarClientError):
    """Exception raised when trying to get a response for a Pulsar async job that has not finished"""

    pass


class PulsarClient(CachingClient):
    """
    Client for the models where we evaluate offline. Since the queries are handled offline, the `PulsarClient` just
    checks if the request/result is cached. We return the result if it's in the cache. Otherwise, we return an error.
    """

    PULSAR_URL: str = os.environ.get('PULSAR_URL', "http://localhost")
    PULSAR_PORT: str = os.environ.get('PULSAR_PORT', "8080")
    INFERENCE_ENDPOINT: str = f"{PULSAR_URL}:{PULSAR_PORT}/generate"
    RETRIEVE_JOB_MAX_WAIT_SECONDS: int = 60

    def convert_to_raw_request(self, request: Request) -> Dict:
        raw_request = {
            "inputs": request.prompt,
            "parameters":{
                "max_new_tokens": request.max_tokens,
                "do_sample": (request.temperature != 0),
                "temperature": request.temperature if request.temperature > 0 else None,
                "top_p": request.top_p if 0 < request.top_p < 1.0 else None,
                "stop": [] if request.stop_sequences is None else request.stop_sequences,
                "details": True,
                "decoder_input_details": request.echo_prompt,
                "truncation": True,
                "model_name": os.environ.get("PULSAR_MODEL_NAME", request.model_engine)
            }
        }

        bos_token = os.environ.get('BOS_TOKEN', '<s>')
        if not request.prompt.startswith(bos_token):
            raw_request['inputs'] = bos_token + raw_request['inputs']
        
        return raw_request

    def __init__(self, cache_config: CacheConfig,  **kargs):
        super().__init__(cache_config=cache_config)
        pass

    def make_request(self, request: Request) -> RequestResult:
        raw_request = self.convert_to_raw_request(request)
        cache_key = CachingClient.make_cache_key(raw_request, request)
        
        def do_it_sync() -> Dict[Any, Any]:

            response = requests.post(PulsarClient.INFERENCE_ENDPOINT, json=raw_request)
            
            try:
                response.raise_for_status()
            except Exception as e:
                raise PulsarClientError(
                    f"Pulsar request failed with {response.status_code}: {response.text}"
                ) from e
            result = response.json()
            if "error" in result:
                error_message = result["error"]
                raise PulsarClientError(f"Pulsar request failed with error: {error_message}")

            text = result['generated_text']

            if raw_request["parameters"]["stop"] is not None:
                text = post_processing_text(text, raw_request["parameters"]["stop"])

            # print("[[parameters]]", raw_request['parameters'])
            # print('[[in]]:', raw_request['inputs'])
            # print('[[out]]:', text)
            
            return {'choices': [
                {
                    'finish_reason': 'length',
                    'text': text,
                    'tokens': [token['text'] for token in result['details']['tokens']]        
                }],
                'details': result['details'],
                'request_id': '0'
            }
                
        try:
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it_sync))
        except Exception as error:
            return RequestResult(
                success=False,
                cached=False,
                error=str(error),
                completions=[],
                embedding=[],
            )
        
        # Expect the result to be structured the same way as a response from OpenAI API.
        completions: List[Sequence] = []
        for raw_completion in response["choices"]:
            sequence_logprob = 0
            tokens: List[Token] = []

            # TODO: take this out when "logprobs" is supported properly in batch/offline mode
            # Currently, token_logprobs is provided in interactive/online mode but it has a different format
            # Waiting for a fix.
            if "logprobs" in raw_completion:
                raw_data = raw_completion["logprobs"]
                for text, logprob, top_logprobs in zip(
                    raw_data["tokens"], raw_data["token_logprobs"], raw_data["top_logprobs"]
                ):
                    # TODO #1654: Check if this is still needed
                    text = cleanup_str(text, "together")
                    tokens.append(Token(text=text, logprob=logprob or 0, top_logprobs=dict(top_logprobs or {})))
                    sequence_logprob += logprob or 0
            else:
                # hack: just make the entire text one token so that something shows up in the frontend
                text = cleanup_str(raw_completion["text"], "together")
                tokens.append(Token(text=text, logprob=0, top_logprobs={}))

            raw_finish_reason: Optional[str] = raw_completion.get("finish_reason")
            finish_reason: Optional[Dict] = {"reason": raw_finish_reason} if raw_finish_reason else None

            completion = Sequence(
                text=cleanup_str(raw_completion["text"], "together"),
                logprob=sequence_logprob,
                tokens=tokens,
                finish_reason=finish_reason,
            )
            completion = truncate_sequence(completion, request)
            completions.append(completion)

        request_time: Union[float, Dict[str, Any]] = response["request_time"]
        if isinstance(request_time, dict):
            batch_performance_metadata: Dict = response["request_time"]
            return RequestResult(
                success=True,
                cached=cached,
                request_time=0,
                completions=completions,
                batch_size=batch_performance_metadata["batch_size"],
                batch_request_time=batch_performance_metadata["batch_time"],
                embedding=[],
            )
        else:
            return RequestResult(
                success=True,
                cached=cached,
                request_time=response["raw_compute_time"] if "raw_compute_time" in response else request_time,
                completions=completions,
                embedding=[],
            )

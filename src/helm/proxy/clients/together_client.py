from typing import List, Dict, Any, Optional, Union
import requests

from helm.common.cache import Cache, CacheConfig
from helm.common.request import Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)
from .client import Client, wrap_request_time, truncate_sequence


MODEL_ALIASES = {
    "flan-t5-xxl": "flan-t5-xxl-hf",
    "h3-2.7b": "h3-2.7b-h3",
    "gpt-jt-x-6b-v1.1": "Together-gpt-JT-6B-v1.1-X",
    "stablelm-base-alpha-7b-fix2": "stablelm-base-alpha-7b",
    "red-pajama-600B-tokens-fix": "red-pajama-600B-tokens",
    "oasst-sft-7-llama-30b-fix": "oasst-sft-7-llama-30b",
    "bloom-ock-dolly-oasst1-hf": "bloom-ock-dolly-oasst1",
    "hb-150m-fix": "hb-150m",
}
# MODEL_ALIASES = {"h3-2.7b": "h3-2.7b-h3", "gpt-neoxt-chat-base-20b": "gpt-neoxt-chat-20b-v0.15-chatml"}
"""Together model name aliases.

HELM users use a shorter model name (e.g. together/flan-t5-xxl)
whereas the Together client sends and caches requests using
a longer model name that is suffixed with the implementation framework
(e.g. flan-t5-xxl-hf). This allows trackcing exactly which
implementation was used in the cached results, since some results may
be different depending on the implementation (e.g. efficiency metrics).
This also allows future migration of results in the case of changes of
available implementations on Together."""


def fix_text(x: str, model: str) -> str:
    """Fix text that comes back from the API."""
    x = x.replace("▁", " ")
    return x


class TogetherClient(Client):
    """
    Client for the models where we evaluate offline. Since the queries are handled offline, the `TogetherClient` just
    checks if the request/result is cached. We return the result if it's in the cache. Otherwise, we return an error.
    """

    INFERENCE_ENDPOINT: str = "https://staging.together.xyz/inference"

    @staticmethod
    def convert_to_raw_request(request: Request) -> Dict:
        # Following the examples from https://github.com/togethercomputer/open-models-api
        req = {
            "request_type": "language-model-inference",
            # "model": MODEL_ALIASES.get(request.model_engine, request.model_engine),
            "model": request.model_engine,
            "prompt": request.prompt,
            "temperature": request.temperature,
            "n": request.num_completions,
            "max_tokens": request.max_tokens,
            "best_of": request.top_k_per_token,
            "logprobs": request.top_k_per_token,
            "stop": request.stop_sequences or None,
            "echo": request.echo_prompt,
            "top_p": request.top_p,
            "stream_tokens": False,
        }
        
        if 'bloom' in request.model_engine or 'mpt' in request.model_engine:
            req['logprobs'] = 0
        
        return req

    def __init__(self, cache_config: CacheConfig, api_key: Optional[str] = None):
        # TODO: the endpoint currently doesn't require an API key. When an API key is not specified
        #       in credentials.conf, we rely on offline evaluation only.
        self.api_key: Optional[
            str
        ] = "Bearer 568af9d9b24c8910c6b30b07f1d10a9c01c775252580befd3d882c5a85d889b4"  # api_key
        self.cache = Cache(cache_config)

    def make_request(self, request: Request) -> RequestResult:
        raw_request = TogetherClient.convert_to_raw_request(request)
        cache_key: Dict = Client.make_cache_key(raw_request, request)
        
        # raw_request['model'] = MODEL_ALIASES.get(request.model_engine, request.model_engine)

        try:

            def do_it():
                raw_request['model'] = MODEL_ALIASES.get(request.model_engine, request.model_engine)
                print(raw_request)
                result = requests.post(
                    TogetherClient.INFERENCE_ENDPOINT,
                    json=raw_request,
                    headers={"Authorization": self.api_key}
                ).json()
                
                try:
                    assert "output" in result, f"Invalid response: {result}"
                    assert "error" not in result["output"], f"Invalid response: {result}"
                except Exception as e:
                    
                    print('!?!?!?!?!?!?!? try api')
                    
                    result = requests.post(
                        "https://api.together.xyz/inference",
                        json=raw_request,
                        headers={"Authorization": self.api_key}
                    ).json()
                    
                assert "output" in result, f"Invalid response: {result}"
                assert "error" not in result["output"], f"Invalid response: {result}"
                
                return result["output"]

            def fail():
                raise RuntimeError(
                    f"The result has not been uploaded to the cache for the following request: {cache_key}"
                )

            response, cached = self.cache.get(cache_key, wrap_request_time(do_it if self.api_key else fail))
        except RuntimeError as e:
            print(e)
            error: str = f"TogetherClient error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        # Expect the result to be structured the same way as a response from OpenAI API.
        completions: List[Sequence] = []
        
        if 'error' in response:
            print(raw_request)
            print('!!!!!!!!!!!!!!!!!!')
        for raw_completion in response["choices"]:
            sequence_logprob = 0
            tokens: List[Token] = []

            # TODO: take this out when "logprobs" is supported properly in batch/offline mode
            # Currently, token_logprobs is provided in interactive/online mode but it has a different format
            # Waiting for a fix.
            if "logprobs" in raw_completion and raw_completion['logprobs']['top_logprobs'] is not None:
                raw_data = raw_completion["logprobs"]
                for text, logprob, top_logprobs in zip(
                    raw_data["tokens"], raw_data["token_logprobs"], raw_data["top_logprobs"]
                ):
                    text = fix_text(text, request.model)
                    tokens.append(Token(text=text, logprob=logprob or 0, top_logprobs=dict(top_logprobs or {})))
                    sequence_logprob += logprob or 0
            else:
                # hack: just make the entire text one token so that something shows up in the frontend
                text = fix_text(raw_completion["text"], request.model)
                tokens.append(Token(text=text, logprob=0, top_logprobs={}))

            completion = Sequence(
                text=fix_text(raw_completion["text"], request.model),
                logprob=sequence_logprob,
                tokens=tokens,
                finish_reason={"reason": raw_completion["finish_reason"]},
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

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        raise NotImplementedError("Use the HuggingFaceClient to tokenize.")

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        raise NotImplementedError("Use the HuggingFaceClient to decode.")

from copy import deepcopy
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
)
from typing import Any, Dict, List, Optional, TypedDict

from helm.common.cache import CacheConfig
from helm.common.hierarchical_logger import htrack_block, hlog
from helm.common.request import (
    wrap_request_time,
    EMBEDDING_UNAVAILABLE_REQUEST_RESULT,
    Request,
    RequestResult,
    Sequence,
    Token,
)
from .client import CachingClient, truncate_sequence
from helm.proxy.tokenizers.huggingface_tokenizer import HuggingFaceTokenizer, WrappedPreTrainedTokenizer, resolve_alias
from threading import Lock
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


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


class StopAtSpecificTokenCriteria(StoppingCriteria):
    def __init__(self, stop_sequence: List[int]):
        super().__init__()
        self.stop_sequence = stop_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Create a tensor from the stop_sequence
        stop_sequence_tensor = torch.tensor(self.stop_sequence, device=input_ids.device, dtype=input_ids.dtype)

        # Check if the current sequence ends with the stop_sequence
        current_sequence = input_ids[:, -len(self.stop_sequence) :]
        return bool(torch.all(current_sequence == stop_sequence_tensor).item())


class HuggingFaceRequest(TypedDict):
    """Data passed between make_request and serve_request. Used as the cache key."""

    engine: str
    prompt: str
    temperature: float
    num_return_sequences: int
    max_new_tokens: int
    top_p: float
    echo_prompt: bool
    top_k_per_token: int
    stop_sequences: List

from datasets import load_dataset
from quanto import Calibration, freeze, qfloat8, qint4, qint8, quantize
@torch.no_grad()
def calibrate_func(model, tokenizer, batch_size, batches):
    samples = batch_size * batches
    cal_dataset = load_dataset("lambada", split=["validation"])[0]
    model.eval()
    total = 0
    for batch in cal_dataset.iter(batch_size=batch_size):
        inputs = tokenizer(batch["text"], return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        model(input_ids, attention_mask=attention_mask)
        total += input_ids.size(0)
        if total >= samples:
            break

class HuggingFaceServer:
    """A thin wrapper around a Hugging Face AutoModelForCausalLM for HuggingFaceClient to call."""

    def __init__(self, pretrained_model_name_or_path: str, **kwargs):
        if torch.cuda.is_available():
            hlog("CUDA is available, initializing with a GPU...")
            self.device: str = "cuda:0"
        else:
            self.device = "cpu"
        with htrack_block(f"Loading Hugging Face model {pretrained_model_name_or_path}"):
            # WARNING this may fail if your GPU does not have enough memory
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=True, torch_dtype=torch.float16, device_map='auto', **kwargs
            )
        with htrack_block(f"Loading Hugging Face tokenizer for model {pretrained_model_name_or_path}"):
            self.wrapped_tokenizer: WrappedPreTrainedTokenizer  = HuggingFaceTokenizer.create_tokenizer(pretrained_model_name_or_path, **kwargs)
        #quantize(self.model, weights=qfloat8)
        quantize(self.model, weights=qfloat8, activations=qfloat8)
        tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1')
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        batch_size = 32
        batches = 4
        # samples = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:batch_size*batches]")
        # text_data = samples['text']
        # input_ids = tokenizer(text_data, return_tensors='pt', padding=True, truncation=True).input_ids.to(self.device)
        torch.backends.cudnn.enable = True
        torch.backends.cudnn.benchmark = True
        with Calibration(momentum=0.9):
            calibrate_func(self.model, tokenizer, batch_size, batches)
        freeze(self.model)



    def serve_request(self, raw_request: HuggingFaceRequest):
        with self.wrapped_tokenizer as tokenizer:
            encoded_input = tokenizer(raw_request["prompt"], return_tensors="pt", return_token_type_ids=False).to(
                self.device
            )
        top_k_per_token: int = raw_request["top_k_per_token"]
        stopping_criteria: Optional[StoppingCriteriaList] = None
        optional_args = {}
        if len(raw_request["stop_sequences"]) > 0:
            with self.wrapped_tokenizer as tokenizer:
                stop_sequence_ids = tokenizer(
                    raw_request["stop_sequences"], return_token_type_ids=False, add_special_tokens=False
                )
            if len(stop_sequence_ids.input_ids) == 1 and len(stop_sequence_ids.input_ids[0]) == 1:
                optional_args["eos_token_id"] = stop_sequence_ids.input_ids[0][0]
            else:
                stopping_criteria = StoppingCriteriaList()
                for stop_sequence_input_ids in stop_sequence_ids.input_ids:
                    stopping_criteria.append(StopAtSpecificTokenCriteria(stop_sequence=stop_sequence_input_ids))

        # Check if we need to compute the perplexity of the prompt (#1497)
        compute_logprobs_only = (
            raw_request["max_new_tokens"] == 0
            and raw_request["num_return_sequences"] == 1
            and raw_request["echo_prompt"]
        )

        # Use HuggingFace's `generate` method.
        if compute_logprobs_only:
            with torch.no_grad():
                output = self.model(encoded_input["input_ids"])
            sequences = encoded_input["input_ids"]
            scores = output.logits
        else:
            if raw_request["temperature"] > 0.001:
                do_sample = True
            else:
                do_sample = False
                raw_request["temperature"] = 1.0

            output = self.model.generate(
                **encoded_input,
                temperature=raw_request["temperature"],
                num_return_sequences=raw_request["num_return_sequences"],
                max_new_tokens=raw_request["max_new_tokens"],
                top_p=raw_request["top_p"],
                do_sample=do_sample,
                return_dict_in_generate=True,
                output_scores=True,
                **optional_args,
                stopping_criteria=stopping_criteria,
            )
            sequences = output.sequences
            scores = output.scores

        prompt_tokens_logprobs = []
        prompt_tokens_top_logprobs_dicts: List[Dict] = []
        if compute_logprobs_only:
            # Append the logprob of the first token of the prompt.
            prompt_tokens_logprobs.append(0.0)
            prompt_tokens_top_logprobs_dicts.append({})

            # Compute logprobs of prompt tokens.
            for completion_id in range(raw_request["num_return_sequences"]):
                for i in range(len(sequences[completion_id]) - 1):
                    logprobs = torch.nn.functional.log_softmax(scores[completion_id][i], dim=0)
                    topk_logprobs = torch.topk(logprobs, k=top_k_per_token)
                    with self.wrapped_tokenizer as tokenizer:
                        prompt_tokens_top_logprobs_dicts.append(
                            {
                                tokenizer.convert_ids_to_tokens(k.item()): v.item()
                                for (k, v) in zip(topk_logprobs.indices, topk_logprobs.values)
                            }
                        )
                    prompt_tokens_logprobs.append(logprobs[sequences[completion_id][i + 1]].item())

        # Compute logprobs of generated tokens for each completed sequence.
        all_generated_tokens_logprobs = []
        all_generated_tokens_top_logprobs_dicts = []
        for completion_id in range(raw_request["num_return_sequences"]):
            generated_tokens_logprobs = []
            generated_tokens_top_logprobs_dicts = []
            for i in range(len(sequences[completion_id]) - len(encoded_input.input_ids[0])):
                logprobs = torch.nn.functional.log_softmax(scores[i][completion_id], dim=0)
                # Get top tokens in terms of log probability.
                topk_logprobs = torch.topk(logprobs, k=top_k_per_token)
                with self.wrapped_tokenizer as tokenizer:
                    generated_tokens_top_logprobs_dicts.append(
                        {
                            tokenizer.convert_ids_to_tokens(k.item()): v.item()
                            for (k, v) in zip(topk_logprobs.indices, topk_logprobs.values)
                        }
                    )
                # Get log probability of chosen token.
                j = i + len(encoded_input.input_ids[0])
                generated_tokens_logprobs.append(logprobs[sequences[completion_id][j]].item())
            all_generated_tokens_logprobs.append(generated_tokens_logprobs)
            all_generated_tokens_top_logprobs_dicts.append(generated_tokens_top_logprobs_dicts)

        # Remove prompt from the start of each sequence if echo_prompt is False.
        if not raw_request["echo_prompt"]:
            sequences = [sequence[len(encoded_input.input_ids[0]) :] for sequence in sequences]

        with self.wrapped_tokenizer as tokenizer:
            all_tokens = [[tokenizer.decode(token) for token in sequence_tokens] for sequence_tokens in sequences]
            all_decoded_text = tokenizer.batch_decode(sequences)

        # post-process
        with self.wrapped_tokenizer as tokenizer:
            if raw_request["stop_sequences"] is None:
                stop_sequences = [tokenizer.eos_token]
            else:
                stop_sequences = raw_request["stop_sequences"] + [tokenizer.eos_token]
            all_decoded_text = [
                post_processing_text(decoded_text, stop_sequences) for decoded_text in all_decoded_text
            ]

        # print('in:', raw_request["prompt"])
        # print('out:', all_decoded_text[0])

        completions = []
        for decoded_text, tokens, generated_tokens_logprobs, generated_tokens_top_logprobs_dicts in zip(
            all_decoded_text, all_tokens, all_generated_tokens_logprobs, all_generated_tokens_top_logprobs_dicts
        ):
            completions.append(
                {
                    "text": decoded_text,
                    "tokens": tokens,
                    "logprobs": generated_tokens_logprobs,
                    "top_logprobs_dicts": generated_tokens_top_logprobs_dicts,
                    "prompt_logprobs": prompt_tokens_logprobs,
                    "prompt_top_logprobs_dicts": prompt_tokens_top_logprobs_dicts,
                }
            )

        return {"completions": completions, "input_length": len(encoded_input.input_ids[0])}


class HuggingFaceServerFactory:
    """A factory that creates and caches HuggingFaceServer objects."""

    _servers: Dict[str, HuggingFaceServer] = {}
    _servers_lock: Lock = Lock()

    @staticmethod
    def get_server(helm_model_name: str, pretrained_model_name_or_path: str, **kwargs) -> Any:
        """
        Checks if the desired HuggingFaceModel is cached. Creates the HuggingFaceModel if it's not cached.
        Returns the HuggingFaceModel.
        """
        with HuggingFaceServerFactory._servers_lock:
            if helm_model_name not in HuggingFaceServerFactory._servers:
                with htrack_block(
                    f"Loading {pretrained_model_name_or_path} (kwargs={kwargs}) "
                    f"for HELM model {helm_model_name} with Hugging Face Transformers"
                ):
                    HuggingFaceServerFactory._servers[helm_model_name] = HuggingFaceServer(
                        pretrained_model_name_or_path, **kwargs
                    )

        return HuggingFaceServerFactory._servers[helm_model_name]


TORCH_DTYPE_KEY = "torch_dtype"
TORCH_DTYPE_VALUE_PREFIX = "torch."


def _process_huggingface_client_kwargs(raw_kwargs: Dict[str, Any]):
    """Process the kwargs for HuggingFaceClient.

    The kwargs passed to HuggingFaceClient will eventually be passed to AutoModel.from_pretrained().
    Since the kwargs from HuggingFaceClient may be derived from configuration YAML,
    they may contain primitive types instead of the unserializable types that
    AutoModel.from_pretrained() expects (e.g. torch_dtype). This function converts values of
    primitive types to values of the unserializable types."""
    processed_kwargs = deepcopy(raw_kwargs)

    # Convert torch_dtype string value to actual dtypes
    # e.g. the string "torch.bfloat16" is converted to torch.bfloat16
    torch_dtype = processed_kwargs.get(TORCH_DTYPE_KEY)
    if torch_dtype and isinstance(torch_dtype, str):
        if not torch_dtype.startswith(TORCH_DTYPE_VALUE_PREFIX):
            raise ValueError(f'Unknown dtype "{torch_dtype}"; expected a string such as "torch.bfloat16"')
        processed_kwargs[TORCH_DTYPE_KEY] = getattr(torch, torch_dtype[len(TORCH_DTYPE_VALUE_PREFIX) :])

    return processed_kwargs


class HuggingFaceClient(CachingClient):
    def __init__(self, cache_config: CacheConfig, pretrained_model_name_or_path: Optional[str] = None, **kwargs):
        super().__init__(cache_config=cache_config)
        self._pretrained_model_name_or_path = pretrained_model_name_or_path
        self._kwargs = _process_huggingface_client_kwargs(kwargs)

    def make_request(self, request: Request) -> RequestResult:
        # Embedding not supported for this model
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

        raw_request: HuggingFaceRequest = {
            "engine": request.model_engine,
            "prompt": request.prompt,
            "temperature": 1e-7 if request.temperature == 0 else request.temperature,
            "num_return_sequences": request.num_completions,
            "max_new_tokens": request.max_tokens,
            "top_p": request.top_p,
            "echo_prompt": request.echo_prompt,
            "top_k_per_token": request.top_k_per_token,
            "stop_sequences": request.stop_sequences,
        }

        pretrained_model_name_or_path: str
        if self._pretrained_model_name_or_path:
            pretrained_model_name_or_path = self._pretrained_model_name_or_path
        else:
            pretrained_model_name_or_path = resolve_alias(request.model_deployment)
        huggingface_model: HuggingFaceServer = HuggingFaceServerFactory.get_server(
            helm_model_name=request.model_deployment,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            **self._kwargs,
        )

        try:

            def do_it():
                return huggingface_model.serve_request(raw_request)

            cache_key = CachingClient.make_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:  # Do something if error is encountered.
            error: str = f"HuggingFace error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions = []
        for raw_completion in response["completions"]:
            sequence_logprob: float = 0
            tokens: List[Token] = []

            if request.echo_prompt:
                # Add prompt to list of generated tokens.
                generated_tokens = raw_completion["tokens"][response["input_length"] :]
                if raw_completion.get("prompt_logprobs") and raw_completion.get("prompt_top_logprobs_dicts"):
                    for token_text, logprob, top_logprobs_dict in zip(
                        raw_completion["tokens"][: response["input_length"]],
                        raw_completion["prompt_logprobs"][: response["input_length"]],
                        raw_completion["prompt_top_logprobs_dicts"][: response["input_length"]],
                    ):
                        tokens.append(Token(text=token_text, logprob=logprob, top_logprobs=top_logprobs_dict))
                        sequence_logprob += logprob
                else:
                    for token_text in raw_completion["tokens"][: response["input_length"]]:
                        tokens.append(Token(text=token_text, logprob=0.0, top_logprobs={}))

            else:
                generated_tokens = raw_completion["tokens"]

            # Compute logprob for the entire sequence.
            for token_text, logprob, top_logprobs_dict in zip(
                generated_tokens, raw_completion["logprobs"], raw_completion["top_logprobs_dicts"]
            ):
                tokens.append(Token(text=token_text, logprob=logprob, top_logprobs=top_logprobs_dict))
                sequence_logprob += logprob

            completion = Sequence(text=raw_completion["text"], logprob=sequence_logprob, tokens=tokens)
            completion = truncate_sequence(completion, request)
            completions.append(completion)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )

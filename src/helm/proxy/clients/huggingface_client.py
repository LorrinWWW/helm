from copy import deepcopy
import torch
from dataclasses import asdict
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Any, Dict, List

from helm.common.cache import Cache, CacheConfig
from helm.common.hierarchical_logger import htrack_block, hlog
from helm.common.request import EMBEDDING_UNAVAILABLE_REQUEST_RESULT, Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
    TokenizationToken,
)
from .client import Client, wrap_request_time, truncate_sequence, cleanup_tokens
from .huggingface_tokenizer import HuggingFaceTokenizers
from helm.proxy.clients.huggingface_model_registry import (
    get_huggingface_model_config,
    HuggingFaceModelConfig,
    HuggingFaceHubModelConfig,
    HuggingFaceLocalModelConfig,
)
import threading
from threading import Lock


# Map of HELM model name to Hugging Face Hub model name where they differ.
_KNOWN_MODEL_ALIASES: Dict[str, str] = {
    "huggingface/gpt2": "gpt2",
    "huggingface/starcoder": "bigcode/starcoder",
}


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


class HuggingFaceServer:
    def __init__(self, model_config: HuggingFaceModelConfig):

        self.available_gpus = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        self.idx_thread_lock = threading.Lock()
        self.idx_gpu = 0
        self.gpu_thread_locks = [
            threading.Lock() for _ in self.available_gpus
        ]
        
        if torch.cuda.is_available():
            hlog("CUDA is available, initializing with a GPU...")
            self.device: str = "cuda:0"
        else:
            self.device = "cpu"
            
        model_kwargs = {}
        # If the HuggingFace model is stored locally, it will have a path defined and we should load it from there.
        # Otherwise, download it from the HuggingFace hub by passing in its identifier.
        if isinstance(model_config, HuggingFaceLocalModelConfig):
            model_name = model_config.path
        elif isinstance(model_config, HuggingFaceHubModelConfig):
            model_name = model_config.model_id
            if model_config.revision:
                model_kwargs["revision"] = model_config.revision
        else:
            raise Exception(f"Unknown type of model_config: {model_config}")
        with htrack_block(f"Loading Hugging Face model for config {model_config}"):
            # WARNING this may fail if your GPU does not have enough memory

            model_name_map = {
                'huggingface/mistral-7b-v0.1': '/work/jue_checkpoints/Mistral-7B-v0.1',
                # 'huggingface/stride-hyena-mistral-7b': 'local_models/StripedHyena-Hessian-Nous-7B',
                'huggingface/stride-hyena-mistral-nous-7b': 'local_models/StripedHyena-Nous-OpenHermes2_5-7B-run5-step1955',
                'huggingface/stride-hyena-mistral-jt-7b-800': 'local_models/StripedHyena-Hessian-Nous-JT-7B',
                'huggingface/stride-hyena-mistral-jt-7b-1200': 'local_models/StripedHyena-Hessian-Nous-JT-7B-1200',
                'huggingface/stride-hyena-mistral-jt-7b-1600': 'local_models/StripedHyena-Hessian-Nous-JT-7B-1600',
                'huggingface/stride-hyena-mistral-jt-7b-2000': 'local_models/StripedHyena-Hessian-Nous-JT-7B-2000',
                'huggingface/stride-hyena-mistral-jt-7b-small-bsz-2000': 'local_models/StripedHyena-Hessian-Nous-JT-7B-small-bsz-2000',
                'huggingface/stride-hyena-mistral-more-wiki-7b-4000': 'local_models/StripedHyena-Hessian-Nous-Wiki-7B-4000',
                'huggingface/stride-hyena-mistral-more-ni-7b-400': 'local_models/StripedHyena-Hessian-Nous-more-ni-7B-400',
                'huggingface/stride-hyena-mistral-kqa-pro-7b-3000': 'local_models/StripedHyena-Hessian-Nous-kqa-pro-7B-3000',
                'huggingface/stride-hyena-mistral-lc-quad2-7b-3000': 'local_models/StripedHyena-Hessian-Nous-lc-quad2-7B-3000',
                'huggingface/stride-hyena-mistral-complex-web-questions-7b-3000': 'local_models/StripedHyena-Hessian-Nous-complex-web-questions-7B-3000',
                'huggingface/stride-hyena-mistral-more-wiki-ni-7b-1200': 'local_models/StripedHyena-Hessian-Nous-more-wiki-ni-7B-1200',
                'huggingface/stride-hyena-mistral-more-wiki-ni-7b-3000': 'local_models/StripedHyena-Hessian-Nous-more-wiki-ni-7B-3000',
                'huggingface/stride-hyena-mistral-more-wiki-ni-7b-6000': 'local_models/StripedHyena-Hessian-Nous-more-wiki-ni-7B-6000',
                'huggingface/stride-hyena-mistral-nq-7b-1000': 'local_models/StripedHyena-Hessian-Nous-nq-7B-1000',
                'huggingface/stride-hyena-mistral-retrieve-nq-7b-1000': 'local_models/StripedHyena-Hessian-Nous-retrieve-nq-7B-1000',
                'huggingface/stride-hyena-mistral-context-nq-7b-1000': 'local_models/StripedHyena-Hessian-Nous-context-nq-7B-1000',
                'huggingface/llama-2-70b': '/work/jue_checkpoints/llama-2-70b',
                'huggingface/llama-65b': '/work/jue_checkpoints/llama-65b',
                'huggingface/stride-hyena-mistral-nous-rc1-2000-7b': '/work/jue_checkpoints/StripedHyena-Nous-OpenHermes2_5-7B-rc1-step2000',
                'huggingface/stride-hyena-mistral-base-7b': '/work/jue_checkpoints/StripedHyena-Hessian-7B',
            }
            
            if '70b' in model_name.lower() or '65b' in model_name.lower():
                print('large model, use accelerate')
                from accelerate import init_empty_weights,load_checkpoint_and_dispatch
                with init_empty_weights():
                    model_name = model_name_map.get(model_name, model_name)
                    config = AutoConfig.from_pretrained(model_name)
                    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
                    model = load_checkpoint_and_dispatch(
                        model, model_name, device_map="auto", no_split_module_classes=["GPTNeoXLayer", "DecoderLayer", "LlamaDecoderLayer", "MPTBlock", "CodeGenBlock"],
                    ).eval()
                    self.model = model
                    self.models = [model]
            else:
                torch.nn.Linear.reset_parameters = lambda x: None
                model_name = model_name_map.get(model_name, model_name)
                # model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, **model_kwargs).eval()

                self.models = []
                for device in self.available_gpus:
                    torch.cuda.set_device(device)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, **model_kwargs
                    ).eval().to(device)
                    self.models.append(model)
                self.model = self.models[0]

                print(f"{len(self.models)} models have been initialized.")
                
        with htrack_block(f"Loading Hugging Face tokenizer model for config {model_config}"):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **model_kwargs)

    def serve_request(self, raw_request: Dict[str, Any]):

        # Select a model for the current thread
        model = self.model
        gpu_thread_lock = self.gpu_thread_locks[0]
        if len(self.models) > 1:
            model = self.models[self.idx_gpu]
            gpu_thread_lock = self.gpu_thread_locks[self.idx_gpu]
            with self.idx_thread_lock:
                self.idx_gpu += 1
                if self.idx_gpu >= len(self.models):
                    self.idx_gpu = 0
        
        encoded_input = self.tokenizer(raw_request["prompt"], return_tensors="pt", return_token_type_ids=False).to(
            model.device
        )
        raw_request = deepcopy(raw_request)
        if 'do_sample' not in raw_request:
            raw_request["do_sample"] = True
        if 'version' in raw_request:
            raw_request.pop('version')
        raw_request["return_dict_in_generate"] = True
        raw_request["output_scores"] = True
        top_k_per_token: int = raw_request["top_k_per_token"]
        del raw_request["top_k_per_token"]

        # Strip out irrelevant parameters
        relevant_raw_request = {
            key: raw_request[key]
            for key in raw_request
            if key not in ["engine", "prompt", "echo_prompt", "stop_sequences"]
        }

        # Check if we need to compute the perplexity of the prompt (#1497)
        compute_logprobs_only = (
            raw_request["max_new_tokens"] == 0
            and raw_request["num_return_sequences"] == 1
            and raw_request["echo_prompt"]
        )

        # Make sure two threads will not use the same gpu
        with gpu_thread_lock:

            print(f'Using device {model.device}')
            
            # Use HuggingFace's `generate` method.
            if compute_logprobs_only:
                with torch.no_grad():
                    output = model(encoded_input["input_ids"])
                    sequences = encoded_input["input_ids"]
                    scores = output.logits
            else:
                output = model.generate(**encoded_input, **relevant_raw_request)
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
                    logprobs = torch.nn.functional.log_softmax(scores[completion_id][i].float(), dim=0)
                    topk_logprobs = torch.topk(logprobs, k=top_k_per_token)
                    prompt_tokens_top_logprobs_dicts.append(
                        {
                            self.tokenizer.convert_ids_to_tokens(k.item()): v.item()
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
                generated_tokens_top_logprobs_dicts.append(
                    {
                        self.tokenizer.convert_ids_to_tokens(k.item()): v.item()
                        for (k, v) in zip(topk_logprobs.indices, topk_logprobs.values)
                    }
                )
                j = i + len(encoded_input.input_ids[0])
                generated_tokens_logprobs.append(logprobs[sequences[completion_id][j]].item())
            all_generated_tokens_logprobs.append(generated_tokens_logprobs)
            all_generated_tokens_top_logprobs_dicts.append(generated_tokens_top_logprobs_dicts)

        # Remove prompt from the start of each sequence if echo_prompt is False.
        if not raw_request["echo_prompt"]:
            sequences = [sequence[len(encoded_input.input_ids[0]) :] for sequence in sequences]

        all_tokens = [[self.tokenizer.decode(token) for token in sequence_tokens] for sequence_tokens in sequences]
        all_decoded_text = self.tokenizer.batch_decode(sequences)

        all_decoded_text = [post_processing_text(text, raw_request["stop_sequences"] + ['</s>']) for text in all_decoded_text]

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


_servers_lock: Lock = Lock()
_servers: Dict[str, HuggingFaceServer] = {}


def _get_singleton_server(model_config: HuggingFaceModelConfig) -> HuggingFaceServer:
    """Lookup or create a new HuggingFaceServer that will be shared among all threads.

    When --num-threads > 1, multiple threads will attempt to instantiate
    `HuggingFaceServer`s simultaneously. Since we have limited GPU memory, we want to
    just share a single copy of each model we are using. So, this function uses a lock
    to make sure that for each model, only one thread creates a HuggingFaceServer.
    The other threads can share that same server in the global _servers dictionary."""
    global _servers_lock
    global _servers
    with _servers_lock:
        if model_config.model_id not in _servers:
            _servers[model_config.model_id] = HuggingFaceServer(model_config)
    return _servers[model_config.model_id]


class HuggingFaceClient(Client):
    def __init__(self, cache_config: CacheConfig):
        self.cache = Cache(cache_config)
        self.model_server_instances: Dict[str, HuggingFaceServer] = {}

    def get_model_server_instance(self, model: str) -> HuggingFaceServer:
        model_config = get_huggingface_model_config(model)
        # Special-case some models in so that users don't have to enable them with --enable-huggingface-models
        if not model_config:
            if model in _KNOWN_MODEL_ALIASES:
                model_config = HuggingFaceHubModelConfig.from_string(_KNOWN_MODEL_ALIASES[model])
            else:
                model_config = HuggingFaceHubModelConfig.from_string(model)
        return _get_singleton_server(model_config)

    def make_request(self, request: Request) -> RequestResult:
        # Embedding not supported for this model
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

        # Only a single stop sequence is supported as we can only pass in a single value for `eos_token_id`
        if len(request.stop_sequences) > 1:
            raise ValueError("More than one stop sequence is not supported.")

        raw_request = {
            # "engine": request.model_engine,
            "engine": request.model,
            "prompt": request.prompt,
            "temperature": 1 if request.temperature == 0 or request.echo_prompt else request.temperature,
            "num_return_sequences": request.num_completions,
            "max_new_tokens": request.max_tokens,
            "top_p": request.top_p,
            "echo_prompt": request.echo_prompt,
            "top_k_per_token": request.top_k_per_token,
            "stop_sequences": request.stop_sequences,
            "version": "greedy run 2",
        }
        if request.temperature == 0 and request.echo_prompt == False:
            raw_request.pop('temperature')
            raw_request.pop('top_p')
            raw_request['do_sample'] = False
        if request.echo_prompt:
            raw_request['version'] = "echo fixed"

        # Get cached model server instance if possible (to save on model and tokenizer
        # loading times).
        model_server_instance: HuggingFaceServer = self.get_model_server_instance(request.model)

        try:

            def do_it():
                return model_server_instance.serve_request(raw_request)

            cache_key = Client.make_cache_key(raw_request, request)
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

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        tokenizer = HuggingFaceTokenizers.get_tokenizer(request.tokenizer)
        cache_key = asdict(request)

        try:

            def do_it():
                if request.encode:
                    if request.truncation:
                        tokens = tokenizer.encode(
                            request.text,
                            truncation=request.truncation,
                            max_length=request.max_length,
                            add_special_tokens=False,
                        )
                    else:
                        tokens = tokenizer.encode(request.text, add_special_tokens=False)
                else:
                    if "gpt" in request.tokenizer or request.tokenizer in [
                        "bigscience/bloom",
                        "Writer/palmyra-base",
                        "facebook/opt-66b",
                    ]:
                        # These models already handle the "▁" character correctly with the
                        # convert_tokens_to_string method. We prefer to use this method instead
                        # of the hacky cleanup_tokens method below as it might handle cases
                        # we haven't thought of in cleanup_tokens.
                        tokens = [
                            tokenizer.convert_tokens_to_string([token]) for token in tokenizer.tokenize(request.text)
                        ]
                    else:
                        # Tokenizes the text and returns the tokens as a list of strings,
                        # not a list of token objects (otherwise "Hello world" would be"
                        # ["Hello", "▁world"] and not ["Hello", " world"])
                        # We could do this with a simple replace like this:
                        # tokens = [tokenizer.convert_tokens_to_string([i]) for i in tokenizer.tokenize(request.text)]
                        # But this replaces all the "▁" characters by "", which is not what we want.
                        # This would be problematic as tokenize(" Hello", encode=False) would return ["Hello"]
                        # Just like tokenize("Hello", encode=False) would return ["Hello"].
                        tokens = tokenizer.tokenize(request.text)
                        tokens = cleanup_tokens(tokens, request.tokenizer)
                return {"tokens": tokens}

            result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            error: str = f"HuggingFace error: {e}"
            return TokenizationRequestResult(success=False, cached=False, error=error, text="", tokens=[])

        return TokenizationRequestResult(
            success=True,
            cached=cached,
            text=request.text,
            tokens=[TokenizationToken(value) for value in result["tokens"]],
            request_time=result["request_time"],
        )

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        tokenizer = HuggingFaceTokenizers.get_tokenizer(request.tokenizer)
        cache_key = asdict(request)

        try:

            def do_it():
                return {
                    "text": tokenizer.decode(
                        request.tokens, clean_up_tokenization_spaces=request.clean_up_tokenization_spaces, skip_special_tokens=True,
                    )
                }

            hlog()
            
            result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            error: str = f"HuggingFace error: {e}"
            return DecodeRequestResult(success=False, cached=False, error=error, text="")

        return DecodeRequestResult(
            success=True, cached=cached, text=result["text"], request_time=result["request_time"]
        )

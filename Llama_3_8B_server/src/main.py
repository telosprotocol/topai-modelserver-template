from typing import Dict, Any
import logging

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse, Response
from typing_extensions import assert_never

from ray import serve

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              ChatCompletionResponse,
                                              CompletionRequest,
                                              CompletionResponse,
                                              DetokenizeRequest,
                                              DetokenizeResponse,
                                              ErrorResponse,
                                              TokenizeRequest,
                                              TokenizeResponse)

# yapf: enable
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_tokenization import (
    OpenAIServingTokenization)
from vllm.engine.metrics import RayPrometheusStatLogger

from topaisdk.modelservice import ModelService

logger = logging.getLogger("ray.serve")

app = FastAPI()

DEFAULT_VLLM_STAT_SEC = 5


@serve.deployment
@serve.ingress(app)
class VLLMService(ModelService):

    def __init__(self):
        super().__init__()

    def check_health(self):
        if not self._ray_check_health:
            raise RuntimeError(self.failure_message())
        
    def unhealth(self, message: str):
        self._ray_check_health = False
        self._failure_message = message

    def failure_message(self) -> str:
        return self._failure_message
    
    def is_health(self) -> bool:
        return self._ray_check_health

    def reconfigure(self, config: Dict[str, Any]):
        """
        Reconfigure the deployment with new settings.

        This method updates the engine arguments and associated configurations
        for the deployment. It initializes a new instance of AsyncLLMEngine
        based on the provided engine arguments and resets the chat serving state.

        Args:
            config : Dict[str, Any]
                A dictionary containing the configuration settings which include
                engine arguments under the "engine_args" key and optionally the
                response role under the "response_role" key.

        Returns:
            None

        Logs:
            Logs the engine arguments with which the deployment is being started.

        Example:
            ```python
            config = {
                "engine_args": {
                    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                    "max_model_len": 8192,
                    ...
                },
                "response_role": "assistant"
            }
            deployment_instance.reconfigure(config)
            ```
        """
        engine_args = AsyncEngineArgs(**config["engine_args"])
        logger.info(f"Starting with engine args: {engine_args}")

        self.openai_serving_chat = None
        self.openai_serving_tokenization = None
        self.openai_serving_completion = None
        self.response_role = config.get("response_role", "assistant")
        self.engine_args = engine_args
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.engine.add_logger("ray",
            RayPrometheusStatLogger(
                local_interval=DEFAULT_VLLM_STAT_SEC,
                labels=dict(model_name=engine_args.served_model_name[0]),
                max_model_len=engine_args.max_model_len))


    @app.get("/health")
    async def health(self,) -> Response:
        """Health check."""
        await self.engine.check_health()
        if self._ray_check_health:
            return Response(status_code=200)
        else:
            return Response(content=self.failure_message(), status_code=500)

    @app.get("/replica_context")
    def replica_context(self) -> Response:
        return Response(content='{}'.format(serve.get_replica_context()), status_code=200)

    @app.get("/test_replica_failure")
    def test_replica_failure(self, replica: str) -> Response:
        local_replica = serve.get_replica_context().replica_tag
        if local_replica == replica:
            self.unhealth("test_replica set failure")
            return Response(content="success", status_code=200)
        else:
            return Response(content="no match replica:{}".format(local_replica), status_code=200)

    @app.get("/test_consecutive_failure")
    @ModelService.consecutive_failure
    def test_consecutive_failure(self, failure: int) -> Response:
        if failure == 1:
            raise RuntimeError('test_consecutive_failure')
        else:
            return Response(content=str(failure), status_code=200)

    @app.post("/tokenize")
    @ModelService.consecutive_failure
    async def tokenize(self, request: TokenizeRequest):
        if not self.openai_serving_tokenization:
            model_config = await self.engine.get_model_config()
            if self.engine_args.served_model_name is not None:
                served_model_names = self.engine_args.served_model_name
            else:
                served_model_names = [self.engine_args.model]
            self.openai_serving_tokenization = OpenAIServingTokenization(
                async_engine_client=self.engine,
                model_config=model_config,
                served_model_names=served_model_names,
                lora_modules=None,
                chat_template=None,
                request_logger=None,
            )
    
        generator = await self.openai_serving_tokenization.create_tokenize(request)
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(),
                                status_code=generator.code)
        elif isinstance(generator, TokenizeResponse):
            return JSONResponse(content=generator.model_dump())

        assert_never(generator)

    @app.post("/detokenize")
    @ModelService.consecutive_failure
    async def detokenize(self, request: DetokenizeRequest):
        if not self.openai_serving_tokenization:
            model_config = await self.engine.get_model_config()
            if self.engine_args.served_model_name is not None:
                served_model_names = self.engine_args.served_model_name
            else:
                served_model_names = [self.engine_args.model]
            self.openai_serving_tokenization = OpenAIServingTokenization(
                async_engine_client=self.engine,
                model_config=model_config,
                served_model_names=served_model_names,
                lora_modules=None,
                chat_template=None,
                request_logger=None,
            )

        generator = await self.openai_serving_tokenization.create_detokenize(request)
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(),
                                status_code=generator.code)
        elif isinstance(generator, DetokenizeResponse):
            return JSONResponse(content=generator.model_dump())

        assert_never(generator)

    @app.get("/v1/models")
    @ModelService.consecutive_failure
    async def show_available_models(self):
        if not self.openai_serving_completion:
            model_config = await self.engine.get_model_config()
            if self.engine_args.served_model_name is not None:
                served_model_names = self.engine_args.served_model_name
            else:
                served_model_names = [self.engine_args.model]
            self.openai_serving_completion = OpenAIServingCompletion(
                async_engine_client=self.engine,
                model_config=model_config,
                served_model_names=served_model_names,
                lora_modules=None,
                prompt_adapters=None,
                request_logger=None,
            )

        models = await self.openai_serving_completion.show_available_models()
        return JSONResponse(content=models.model_dump())


    @app.post("/v1/completions")
    @ModelService.consecutive_failure
    async def create_completion(self, request: CompletionRequest, raw_request: Request):
        if not self.openai_serving_completion:
            model_config = await self.engine.get_model_config()
            if self.engine_args.served_model_name is not None:
                served_model_names = self.engine_args.served_model_name
            else:
                served_model_names = [self.engine_args.model]
            self.openai_serving_completion = OpenAIServingCompletion(
                async_engine_client=self.engine,
                model_config=model_config,
                served_model_names=served_model_names,
                lora_modules=None,
                prompt_adapters=None,
                request_logger=None,
            )

        logger.info(f"Request: {request}")
    
        generator = await self.openai_serving_completion.create_completion(
            request, raw_request)
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(),
                                status_code=generator.code)
        elif isinstance(generator, CompletionResponse):
            return JSONResponse(content=generator.model_dump())

        return StreamingResponse(content=generator, media_type="text/event-stream")

    @app.post("/v1/chat/completions")
    @ModelService.consecutive_failure
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        """
        Handle the creation of chat completions.

        This endpoint processes incoming chat requests and generates responses
        using the configured language model. If the OpenAIServingChat instance 
        is not initialized, it sets it up with the appropriate configuration 
        from the engine.

        Args:
            request : ChatCompletionRequest
                The incoming chat completion request.
            raw_request : Request
                The raw HTTP request object.

        Returns:
            JSONResponse or StreamingResponse:
                A JSON response containing the chat completion if request.stream is False,
                otherwise a streaming response containing the chat event stream.
        
        Raises:
            Exception
                If any errors occur during the generation of the chat completion, 
                they will be logged and appropriate error responses will be returned.
        """
        if not self.openai_serving_chat:
            model_config = await self.engine.get_model_config()
            if self.engine_args.served_model_name is not None:
                served_model_names = self.engine_args.served_model_name
            else:
                served_model_names = [self.engine_args.model]
            self.openai_serving_chat = OpenAIServingChat(
                async_engine_client=self.engine,
                model_config=model_config,
                served_model_names=served_model_names,
                response_role=self.response_role,
                lora_modules=None,
                chat_template=None,
                prompt_adapters=None,
                request_logger=None,
            )

        logger.info(f"Request: {request}")
        generator = await self.openai_serving_chat.create_chat_completion(
            request=request,
            raw_request=raw_request,
        )
        
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(),
                status_code=generator.code,
            )
        if request.stream:
            return StreamingResponse(
                content=generator,
                media_type="text/event-stream",
            )
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())


vllmservice = VLLMService.bind()

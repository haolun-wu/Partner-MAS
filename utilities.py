import json
import csv
import os
from math import radians, cos, sin, asin, sqrt
from concurrent.futures import ThreadPoolExecutor, as_completed
from project_types import llm_response_format, llm_token_usage
from typing import Optional, Union, Type
from pydantic import BaseModel

class ParallelProcessor:
    """
    Utility class for parallel processing of tasks using ThreadPoolExecutor.
    Supports configurable number of workers and maintains order of results.
    """

    def __init__(self, num_workers: int = 4):
        """
        Initialize the parallel processor.

        Args:
            num_workers (int): Number of worker threads (default: 4)
        """
        self.num_workers = num_workers

    def process_tasks(self, task_func, task_args_list, preserve_order: bool = True):
        """
        Process tasks in parallel using ThreadPoolExecutor.

        Args:
            task_func (callable): Function to execute for each task
            task_args_list (list): List of argument tuples/dicts for each task
            preserve_order (bool): Whether to preserve the order of results (default: True)

        Returns:
            list: Results in the same order as input tasks (if preserve_order=True)
                 or in completion order (if preserve_order=False)
        """
        results = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            if preserve_order:
                # Submit all tasks and maintain order
                future_to_index = {}
                for i, args in enumerate(task_args_list):
                    if isinstance(args, dict):
                        future = executor.submit(task_func, **args)
                    elif isinstance(args, (list, tuple)):
                        future = executor.submit(task_func, *args)
                    else:
                        future = executor.submit(task_func, args)
                    future_to_index[future] = i

                # Create results list with correct size
                results = [None] * len(task_args_list)

                # Collect results in order
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        results[index] = result
                    except Exception as e:
                        print(f"Task {index} failed with error: {e}")
                        results[index] = None
            else:
                # Process in completion order (faster)
                futures = []
                for args in task_args_list:
                    if isinstance(args, dict):
                        future = executor.submit(task_func, **args)
                    elif isinstance(args, (list, tuple)):
                        future = executor.submit(task_func, *args)
                    else:
                        future = executor.submit(task_func, args)
                    futures.append(future)

                # Collect results as they complete
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"Task failed with error: {e}")
                        results.append(None)

        return results

    def process_with_callback(self, task_func, task_args_list, callback_func=None):
        """
        Process tasks in parallel with optional callback for each completion.

        Args:
            task_func (callable): Function to execute for each task
            task_args_list (list): List of argument tuples/dicts for each task
            callback_func (callable): Optional function to call with (index, result) for each completion

        Returns:
            list: Results in original order
        """
        results = [None] * len(task_args_list)

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_index = {}
            for i, args in enumerate(task_args_list):
                if isinstance(args, dict):
                    future = executor.submit(task_func, **args)
                elif isinstance(args, (list, tuple)):
                    future = executor.submit(task_func, *args)
                else:
                    future = executor.submit(task_func, args)
                future_to_index[future] = i

            # Process completions
            completed_count = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                completed_count += 1
                try:
                    result = future.result()
                    results[index] = result
                    if callback_func:
                        callback_func(index, result, completed_count, len(task_args_list))
                except Exception as e:
                    print(f"Task {index} failed with error: {e}")
                    results[index] = None
                    if callback_func:
                        callback_func(index, None, completed_count, len(task_args_list))

        return results

class openai_client:
    """
    OpenAI API client with token usage tracking and parallel call support.
    Supports GPT-5 series, GPT-4o, GPT-4o-mini, GPT-4.1-mini, and GPT-4.1-nano models with cost calculation.
    Uses the new Response API with advanced parameters like verbosity and reasoning effort.
    """

    def __init__(self, secret_path: str):
        """
        Initialize the OpenAI API client.

        Args:
            secret_path (str): Path to the file containing the API key
        """
        self.api_key = self._read_api_key(secret_path)
        self.test_mode = (self.api_key == "test-key-for-local-testing-only")

        if not self.test_mode:
            import openai as openai_lib
            self.client = openai_lib.OpenAI(api_key=self.api_key)
        else:
            print("🧪 Running in TEST MODE - generating mock responses")
            self.client = None

    def _model_supports_temperature(self, model: str) -> bool:
        """
        Determine if a model supports the temperature parameter.

        Args:
            model (str): The model name to check

        Returns:
            bool: True if model supports temperature, False otherwise

        Rules:
        - GPT-3 series: Support temperature (gpt-3.5-turbo, etc.)
        - GPT-4 series (non-reasoning): Support temperature (gpt-4, gpt-4o, gpt-4-turbo, etc.)
        - GPT-5 series (reasoning models): Do NOT support temperature (gpt-5, gpt-5-nano, etc.)
        - O series (reasoning models): Do NOT support temperature (o1, o3, o3-mini, etc.)
        """
        model_lower = model.lower()

        # Reasoning models that do NOT support temperature
        reasoning_model_prefixes = [
            "gpt-5",      # All GPT-5 series (gpt-5, gpt-5-nano, gpt-5-mini)
            "o1",         # O1 series (o1, o1-mini, o1-preview)
            "o3",         # O3 series (o3, o3-mini)
        ]

        # Check if it's a reasoning model
        for prefix in reasoning_model_prefixes:
            if model_lower.startswith(prefix):
                return False

        # Non-reasoning models that DO support temperature
        # GPT-3 series, GPT-4 series (including gpt-4o, gpt-4-turbo, gpt-4.1, etc.)
        non_reasoning_prefixes = [
            "gpt-3",      # GPT-3 series
            "gpt-4",      # GPT-4 series (but not gpt-5!)
        ]

        for prefix in non_reasoning_prefixes:
            if model_lower.startswith(prefix):
                return True

        # Default: assume newer unknown models are reasoning models
        # This is a conservative approach - better to omit temperature than cause API errors
        return False

    def _read_api_key(self, secret_path: str) -> str:
        """
        Read API key from the secret file.

        Args:
            secret_path (str): Path to the file containing the API keys in JSON format

        Returns:
            str: The OpenAI API key

        Raises:
            FileNotFoundError: If the secret file doesn't exist
            ValueError: If the API key is empty or invalid
            KeyError: If OpenAI provider is not found
        """
        try:
            with open(secret_path, 'r') as f:
                api_keys_data = json.load(f)

            if not isinstance(api_keys_data, list):
                raise ValueError("Secret file should contain a list of API providers")

            # Find the OpenAI API key
            for provider in api_keys_data:
                if provider.get("API provider") == "OpenAI":
                    api_key = provider.get("API key", "").strip()
                    if not api_key:
                        raise ValueError("OpenAI API key is empty")
                    return api_key

            raise KeyError("OpenAI provider not found in secret file")

        except FileNotFoundError:
            raise FileNotFoundError(f"Secret file not found: {secret_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in secret file: {e}")
        except Exception as e:
            raise ValueError(f"Error reading API key: {e}")

    def response_completion(self, user_prompt: str, system_prompt: str = None, model: str = "gpt-5-nano", stream: bool = False, verbosity: str = "medium", reasoning_effort: str = "medium", reasoning_summary: str = None, return_full_response: bool = False, call_id: str = None, schema_format: dict = None, pydantic_model: Type[BaseModel] = None, validate_response: bool = True, temperature: float = 0.7, **kwargs):
        """
        Send a chat completion request using the new response API with optional Pydantic validation.
        Supports GPT-5 series, GPT-4o, GPT-4o-mini, and GPT-4.1-mini models.

        Args:
            user_prompt (str): The user's message
            system_prompt (str): The system prompt/instruction (default: None)
            model (str): The model to use (default: "gpt-5-nano")
                        Supported: "gpt-5", "gpt-5-nano", "gpt-4o", "gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano"
            stream (bool): Whether to stream the response (default: False)
            verbosity (str): Response verbosity level - "low", "medium", "high" (default: "medium")
                            - ONLY for GPT-5 series models (gpt-5, gpt-5-mini, gpt-5-nano)
                            - Controls output length and detail without changing the prompt
                            - Ignored for other models (gpt-4o, gpt-4.1 series, o1, o3)
            reasoning_effort (str): Reasoning effort level (default: "medium")
                                   - ONLY for reasoning models: GPT-5 series, o1, o3, o3-mini
                                   - "minimal": Fastest response, minimal reasoning tokens (GPT-5 only)
                                   - "low", "medium", "high": For reasoning models only
                                   - Ignored for regular models (gpt-4o, gpt-4.1 series)
            reasoning_summary (str): Reasoning summary level - "auto", "concise", "detailed" (default: None)
            return_full_response (bool): Whether to return the full response object (default: False)
            call_id (str): Optional identifier for tracking parallel calls (default: None)
            schema_format (dict): JSON schema for structured output (default: None)
                                 Supports standard OpenAI JSON schema format
            pydantic_model (Type[BaseModel]): Pydantic model for response validation (default: None)
                                            If provided, takes precedence over schema_format
            validate_response (bool): Whether to validate response with Pydantic model (default: True)

        Returns:
            llm_response_format or tuple: The llm_response_format object, or (llm_response_format, full_response) if return_full_response=True
                                        If Pydantic validation is used, includes validated data in the response

        Raises:
            Exception: If the API request fails
            ValidationError: If Pydantic validation fails and validate_response is True
        """
        # Handle test mode
        if self.test_mode:
            return self._generate_mock_response(user_prompt, pydantic_model)

        try:

            messages = []

            # Add system message if provided
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })

            # Add user message
            messages.append({
                "role": "user",
                "content": user_prompt
            })

            # Prepare parameters for the API request
            api_params = {
                "model": model,
                "input": messages,
                "stream": stream
            }

            # Dynamic temperature support based on model capabilities
            if self._model_supports_temperature(model):
                api_params["temperature"] = temperature

            # Prepare text parameter with verbosity and format
            text_params = {}

            # Add verbosity parameter ONLY for GPT-5 series models
            gpt5_models = ["gpt-5", "gpt-5-nano", "gpt-5-mini"]
            if any(model.startswith(gpt5) for gpt5 in gpt5_models) and verbosity in ["low", "medium", "high"]:
                text_params["verbosity"] = verbosity

            # Handle Pydantic model or schema format
            effective_schema = None
            if pydantic_model:
                # Convert Pydantic model to OpenAI schema format
                try:
                    from pydantic_schemas import SchemaRegistry
                    effective_schema = SchemaRegistry.convert_to_openai_schema(pydantic_model)
                except ImportError:
                    # Fallback: manual conversion if pydantic_schemas not available
                    schema = pydantic_model.model_json_schema()

                    effective_schema = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": pydantic_model.__name__.lower(),
                            "strict": True,
                            "schema": schema
                        }
                    }
            elif schema_format:
                effective_schema = schema_format

            # Add response format if we have a schema
            if effective_schema:
                if isinstance(effective_schema, dict) and "json_schema" in effective_schema:
                    # If it's the full OpenAI format, extract the schema and name
                    json_schema = effective_schema["json_schema"]
                    schema_name = json_schema.get("name", "structured_output")
                    schema_def = json_schema.get("schema", {})

                    text_params["format"] = {
                        "type": "json_schema",
                        "name": schema_name,  # Required field
                        "schema": schema_def
                    }
                else:
                    # If it's already just the schema, use a default name
                    text_params["format"] = {
                        "type": "json_schema",
                        "name": "structured_output",  # Default name
                        "schema": effective_schema
                    }

            # Set text parameter if any text config is provided
            if text_params:
                api_params["text"] = text_params

            # Add reasoning parameters ONLY for reasoning models (GPT-5 series, o1, o3, o3-mini)
            reasoning_models = ["gpt-5", "gpt-5-nano", "gpt-5-mini", "o1", "o3", "o3-mini"]
            reasoning_params = {}

            if any(model.startswith(rm) for rm in reasoning_models):
                # Support minimal effort for GPT-5 series, and low/medium/high for reasoning models
                if reasoning_effort in ["minimal", "low", "medium", "high"]:
                    reasoning_params["effort"] = reasoning_effort
                if reasoning_summary in ["auto", "concise", "detailed"]:
                    reasoning_params["summary"] = reasoning_summary

            if reasoning_params:
                api_params["reasoning"] = reasoning_params

            # Make the API request using new response API
            response = self.client.responses.create(**api_params)

            if stream:
                # Handle streaming response
                full_response = ""
                for chunk in response:
                    if hasattr(chunk, 'output') and chunk.output:
                        full_response += chunk.output

                # Create llm_response_format for streaming
                token_usage_obj = self._create_token_usage(response.usage if hasattr(response, 'usage') else None)

                # Perform Pydantic validation on streamed response if requested
                validated_data = None
                validation_errors = []

                if pydantic_model and validate_response:
                    try:
                        from pydantic_schemas import validate_response_with_pydantic
                        validation_result = validate_response_with_pydantic(full_response, pydantic_model)

                        if validation_result.is_valid:
                            validated_data = validation_result.parsed_data
                        else:
                            validation_errors = [f"{error.field}: {error.error}" for error in validation_result.errors]
                            if validate_response:
                                error_msg = f"Pydantic validation failed: {'; '.join(validation_errors)}"
                                raise ValueError(error_msg)

                    except ImportError:
                        # Fallback validation
                        try:
                            import json
                            data = json.loads(full_response)
                            validated_data = pydantic_model.model_validate(data).model_dump()
                        except Exception as e:
                            validation_errors = [f"Validation error: {str(e)}"]
                            if validate_response:
                                raise ValueError(f"Pydantic validation failed: {str(e)}")

                response_obj = llm_response_format(
                    text=full_response,
                    usage=token_usage_obj,
                    summary="",
                    error=""
                )

                # Add validated data as custom attribute if validation was successful
                if validated_data:
                    response_obj.validated_data = validated_data
                if validation_errors:
                    response_obj.validation_errors = validation_errors

                if return_full_response:
                    return response_obj, response
                else:
                    return response_obj
            else:
                # Handle non-streaming response
                # Extract the actual response text from the output structure
                response_text = ""
                if hasattr(response, 'output') and response.output:
                    # The output contains messages with text content
                    for output_item in response.output:
                        if hasattr(output_item, 'content') and output_item.content:
                            for content_item in output_item.content:
                                if hasattr(content_item, 'text') and content_item.text:
                                    response_text = content_item.text
                                    break
                        if response_text:
                            break

                # Fallback: return string representation if parsing fails
                if not response_text:
                    response_text = str(response)

                # Extract reasoning summary if available
                reasoning_summary_text = ""
                if hasattr(response, 'output') and response.output:
                    for output_item in response.output:
                        if hasattr(output_item, 'summary') and output_item.summary:
                            for summary_item in output_item.summary:
                                if hasattr(summary_item, 'text') and summary_item.text:
                                    reasoning_summary_text += summary_item.text + "\n"

                # Create token usage object
                token_usage_obj = self._create_token_usage(response.usage)

                # Perform Pydantic validation if requested
                validated_data = None
                validation_errors = []

                if pydantic_model and validate_response:
                    try:
                        from pydantic_schemas import validate_response_with_pydantic
                        validation_result = validate_response_with_pydantic(response_text, pydantic_model)

                        if validation_result.is_valid:
                            validated_data = validation_result.parsed_data
                        else:
                            validation_errors = [f"{error.field}: {error.error}" for error in validation_result.errors]
                            if validate_response:  # Strict validation - raise error
                                error_msg = f"Pydantic validation failed: {'; '.join(validation_errors)}"
                                raise ValueError(error_msg)

                    except ImportError:
                        # Fallback validation without pydantic_schemas
                        try:
                            import json
                            data = json.loads(response_text)
                            validated_data = pydantic_model.model_validate(data).model_dump()
                        except Exception as e:
                            validation_errors = [f"Validation error: {str(e)}"]
                            if validate_response:
                                raise ValueError(f"Pydantic validation failed: {str(e)}")

                # Create llm_response_format object with validation info
                response_obj = llm_response_format(
                    text=response_text,
                    usage=token_usage_obj,
                    summary=reasoning_summary_text.strip(),
                    error=""
                )

                # Add validated data as custom attribute if validation was successful
                if validated_data:
                    response_obj.validated_data = validated_data
                if validation_errors:
                    response_obj.validation_errors = validation_errors

                # Return based on return_full_response parameter
                if return_full_response:
                    return response_obj, response
                else:
                    return response_obj

        except Exception as e:
            # Create error llm_response_format object
            error_token_usage = llm_token_usage(input=0, output=0, reasoning=0, cached=0, total=0)
            error_response = llm_response_format(
                text="ERROR",
                usage=error_token_usage,
                summary="",
                error=str(e)
            )

            if return_full_response:
                return error_response, None
            else:
                return error_response


    def _create_token_usage(self, usage_obj) -> llm_token_usage:
        """
        Create a llm_token_usage dataclass from OpenAI usage object.

        Args:
            usage_obj: OpenAI usage object from response

        Returns:
            llm_token_usage: Token usage dataclass
        """
        if not usage_obj:
            return llm_token_usage(input=0, output=0, reasoning=0, cached=0, total=0)

        # Extract basic token counts - handle both old and new API formats
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0

        if hasattr(usage_obj, 'prompt_tokens'):
            input_tokens = usage_obj.prompt_tokens
        elif hasattr(usage_obj, 'input_tokens'):
            input_tokens = usage_obj.input_tokens

        if hasattr(usage_obj, 'completion_tokens'):
            output_tokens = usage_obj.completion_tokens
        elif hasattr(usage_obj, 'output_tokens'):
            output_tokens = usage_obj.output_tokens

        if hasattr(usage_obj, 'total_tokens'):
            total_tokens = usage_obj.total_tokens

        # Extract detailed token information
        cached_tokens = 0
        reasoning_tokens = 0

        if hasattr(usage_obj, 'prompt_tokens_details'):
            prompt_details = usage_obj.prompt_tokens_details
            if hasattr(prompt_details, 'cached_tokens'):
                cached_tokens = prompt_details.cached_tokens
        elif hasattr(usage_obj, 'input_tokens_details'):
            input_details = usage_obj.input_tokens_details
            if hasattr(input_details, 'cached_tokens'):
                cached_tokens = input_details.cached_tokens

        if hasattr(usage_obj, 'completion_tokens_details'):
            completion_details = usage_obj.completion_tokens_details
            if hasattr(completion_details, 'reasoning_tokens'):
                reasoning_tokens = completion_details.reasoning_tokens
        elif hasattr(usage_obj, 'output_tokens_details'):
            output_details = usage_obj.output_tokens_details
            if hasattr(output_details, 'reasoning_tokens'):
                reasoning_tokens = output_details.reasoning_tokens

        return llm_token_usage(
            input=input_tokens,
            output=output_tokens,
            reasoning=reasoning_tokens,
            cached=cached_tokens,
            total=total_tokens
        )

    def _generate_mock_response(self, user_prompt: str, pydantic_model: Type[BaseModel] = None) -> llm_response_format:
        """
        Generate a mock response for testing purposes.

        Args:
            user_prompt (str): The user's prompt
            pydantic_model (Type[BaseModel]): Expected Pydantic model for structured response

        Returns:
            llm_response_format: Mock response with sample data
        """
        import json
        from project_types import llm_token_usage

        # Create mock token usage
        mock_usage = llm_token_usage(input=100, output=200, reasoning=0, cached=10, total=300)

        # Generate mock structured response based on Pydantic model
        if pydantic_model:
            try:
                # Generate mock data based on the model schema
                mock_data = self._generate_mock_structured_data(pydantic_model)
                mock_text = json.dumps(mock_data, indent=2)

                response = llm_response_format(
                    text=mock_text,
                    usage=mock_usage,
                    summary="Mock response generated for testing",
                    error=""
                )

                # Add validated data
                response.validated_data = mock_data
                return response

            except Exception as e:
                print(f"⚠️ Error generating mock structured data: {e}")

        # Fallback to generic mock response
        mock_text = '{"message": "This is a mock response for testing purposes.", "status": "test_mode"}'

        return llm_response_format(
            text=mock_text,
            usage=mock_usage,
            summary="Generic mock response",
            error=""
        )

    def _generate_mock_structured_data(self, pydantic_model: Type[BaseModel]) -> dict:
        """Generate mock data that conforms to a Pydantic model schema."""
        # Return basic fallback based on model name first (more reliable than schema parsing)
        model_name = pydantic_model.__name__.lower()
        print(f"🔍 DEBUG: Generating mock data for model: {model_name}")

        # Import here to avoid circular dependencies
        try:
            # Use specific mock data first, then fall back to schema
            fallback_data = self._get_model_specific_mock_data(model_name)
            if fallback_data:
                print(f"✅ Using specific mock data for {model_name}")
                return fallback_data

            print(f"⚠️ No specific mock data for {model_name}, trying schema generation...")
            schema = pydantic_model.model_json_schema()
            return self._create_mock_from_schema(schema)
        except Exception as e:
            print(f"Error generating mock data from schema: {e}")
            # Return generic fallback
            return {"mock_response": "Generic mock data for testing"}

    def _get_model_specific_mock_data(self, model_name: str) -> dict:
        """Get specific mock data for known model types."""
        if "planner" in model_name:
            return {
                    "strategic_guidance": "Mock strategic guidance for testing: Focus on industry expertise and geographic proximity.",
                    "agents": [
                        {
                            "name": "Mock Industry Expert",
                            "role": "Industry Specialist",
                            "ability": "Deep expertise in Computer Related sector with 10+ years experience",
                            "profile": "Specializes in evaluating technology companies and assessing market fit. Focuses on portfolio alignment, technical due diligence, and competitive positioning within the technology sector."
                        },
                        {
                            "name": "Mock Geographic Analyst",
                            "role": "Regional Investment Specialist",
                            "ability": "Regional market analysis and local network connections",
                            "profile": "Focuses on Pacific Northwest technology investments and local market dynamics. Evaluates geographic synergies, local market penetration strategies, and regional partnership opportunities."
                        }
                    ]
                }
        elif "specialized" in model_name or "agent" in model_name:
            return {
                    "evaluation_focus": "Mock evaluation focusing on industry alignment and investment track record",
                    "overall_rationale": "Mock rationale: Candidates selected based on technology sector expertise and successful co-investment history.",
                    "ranked_candidates": [
                        {
                            "firm_id": "838",
                            "rank": 1,
                            "alignment_score": 9,
                            "rationale": "Strong track record in technology investments with relevant industry experience"
                        },
                        {
                            "firm_id": "16789",
                            "rank": 2,
                            "alignment_score": 8,
                            "rationale": "Good geographic alignment and proven collaboration history"
                        },
                        {
                            "firm_id": "17414",
                            "rank": 3,
                            "alignment_score": 7,
                            "rationale": "Local expertise and strong network connections in target market"
                        }
                    ]
                }
        elif "supervisor" in model_name:
            return {
                    "selected_candidates": ["838", "16789", "17414"],
                    "rationale": "Mock supervisor selection based on comprehensive analysis of all agent recommendations, prioritizing industry expertise and proven track records."
                }
        else:
            return {"mock_response": "Generic mock data for testing"}

        # Return None if no specific mock data found
        return None

    def _create_mock_from_schema(self, schema: dict) -> dict:
        """Create mock data from a JSON schema."""
        properties = schema.get("properties", {})
        mock_data = {}

        for field_name, field_schema in properties.items():
            field_type = field_schema.get("type", "string")

            if field_type == "string":
                mock_data[field_name] = f"Mock {field_name.replace('_', ' ')}"
            elif field_type == "integer":
                mock_data[field_name] = 42
            elif field_type == "number":
                mock_data[field_name] = 3.14
            elif field_type == "boolean":
                mock_data[field_name] = True
            elif field_type == "array":
                items_schema = field_schema.get("items", {})
                if items_schema.get("type") == "object":
                    mock_data[field_name] = [self._create_mock_from_schema(items_schema)]
                else:
                    mock_data[field_name] = ["mock_item_1", "mock_item_2"]
            elif field_type == "object":
                mock_data[field_name] = self._create_mock_from_schema(field_schema)

        return mock_data


    def calculate_cost(self, usage: llm_token_usage, model: str = "gpt-5-nano") -> float:
        """
        Calculate the cost of an API call based on token usage.
        Accounts for cached tokens (50% discount) and reasoning tokens.

        Args:
            usage (llm_token_usage): Token usage object from llm_response_format
            model (str): The model used (default: "gpt-5-nano")

        Returns:
            float: The cost in USD
        """
        if not usage:
            return 0.0

        # Pricing per 1K tokens (2025 rates from OpenAI)
        pricing = {
            "gpt-5-nano": {"input": 0.00005, "output": 0.0004, "cached_input": 0.000005},
            "gpt-5": {"input": 0.00125, "output": 0.01, "cached_input": 0.000125},
            "gpt-4o": {"input": 0.0025, "output": 0.01, "cached_input": 0.00125},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006, "cached_input": 0.000075},
            "gpt-4.1-mini": {"input": 0.00015, "output": 0.0006, "cached_input": 0.000075},
            "gpt-4.1-nano": {"input": 0.0001, "output": 0.0004, "cached_input": 0.00005}  # $0.10/1M input, $0.40/1M output
        }

        model_pricing = pricing.get(model, {"input": 0.00005, "output": 0.0004, "cached_input": 0.000005})

        # Calculate input costs (accounting for cached tokens)
        uncached_prompt_tokens = usage.input - usage.cached
        cached_tokens = usage.cached

        # Use separate pricing for cached tokens
        uncached_input_cost = (uncached_prompt_tokens / 1000) * model_pricing["input"]
        cached_input_cost = (cached_tokens / 1000) * model_pricing["cached_input"]

        # Calculate output costs (including reasoning tokens as output)
        total_output_tokens = usage.output + usage.reasoning
        output_cost = (total_output_tokens / 1000) * model_pricing["output"]

        total_cost = uncached_input_cost + cached_input_cost + output_cost

        return round(total_cost, 6)

    def test_connection(self) -> bool:
        """
        Test the API connection with a simple request.

        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            # Just test if we can make a basic API call without checking specific response content
            response_obj = self.response_completion(
                user_prompt="Say hello",
                system_prompt="You are a helpful assistant.",
                model="gpt-5-nano"
            )
            # If we get any response without exception and no error, the connection is working
            return response_obj is not None and not response_obj.error and len(response_obj.text.strip()) > 0
        except Exception:
            return False

    def structured_completion(self, user_prompt: str, pydantic_model: Type[BaseModel], system_prompt: str = None, model: str = "gpt-4o-mini", **kwargs) -> llm_response_format:
        """
        Convenience method for structured completion with Pydantic validation.
        Uses models with better structured output support (gpt-4o-mini, gpt-4o).

        Args:
            user_prompt (str): The user's message
            pydantic_model (Type[BaseModel]): Pydantic model for response validation
            system_prompt (str): The system prompt/instruction (default: None)
            model (str): The model to use (default: "gpt-4o-mini")
            **kwargs: Additional arguments to pass to response_completion

        Returns:
            llm_response_format: Response object with validated_data attribute if successful

        Raises:
            ValueError: If Pydantic validation fails
        """
        # Use models known to have good structured output support
        structured_output_models = ["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini", "gpt-5", "gpt-5-nano"]
        if model not in structured_output_models:
            print(f"Warning: Model '{model}' may not support structured outputs optimally. Recommended: {structured_output_models}")

        return self.response_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model=model,
            pydantic_model=pydantic_model,
            validate_response=True,
            **kwargs
        )

    def flexible_completion(self, user_prompt: str, pydantic_model: Type[BaseModel], system_prompt: str = None, model: str = "gpt-4o-mini", max_retries: int = 3, **kwargs) -> llm_response_format:
        """
        Flexible structured completion with retry logic for validation failures.

        Args:
            user_prompt (str): The user's message
            pydantic_model (Type[BaseModel]): Pydantic model for response validation
            system_prompt (str): The system prompt/instruction (default: None)
            model (str): The model to use (default: "gpt-4o-mini")
            max_retries (int): Maximum number of retries on validation failure (default: 3)
            **kwargs: Additional arguments to pass to response_completion

        Returns:
            llm_response_format: Response object with validated_data attribute

        Raises:
            ValueError: If validation fails after all retries
        """
        for attempt in range(max_retries + 1):
            try:
                response = self.response_completion(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    model=model,
                    pydantic_model=pydantic_model,
                    validate_response=True,
                    **kwargs
                )

                # If we get here, validation succeeded
                return response

            except ValueError as e:
                if "Pydantic validation failed" in str(e) and attempt < max_retries:
                    print(f"Validation attempt {attempt + 1} failed, retrying... Error: {str(e)}")
                    continue
                else:
                    # Final attempt failed or non-validation error
                    raise

        # This shouldn't be reached, but just in case
        raise ValueError(f"Failed to get valid response after {max_retries + 1} attempts")


class ProfileExtractor:
    """
    Utility class for extracting and structuring VC investment data from CSV files.
    Converts raw CSV data into structured profiles for lead investors, target companies, and candidate co-investors.
    """

    def __init__(self):
        """Initialize the ProfileExtractor."""
        self.target_profile = ""
        self.lead_profile = ""
        self.candidate_profiles = {}

    def load_from_csv(self, filepath: str) -> tuple:
        """
        Load investor profiles from a CSV file.

        Args:
            filepath (str): Path to the CSV file containing VC investment data

        Returns:
            tuple: (lead_profile, target_profile, candidate_profiles)
                - lead_profile (str): Structured profile text for lead investor
                - target_profile (str): Structured profile text for target company
                - candidate_profiles (dict): Dictionary of candidate profiles {firm_id: profile_text}

        Raises:
            FileNotFoundError: If the CSV file doesn't exist
            ValueError: If required data fields are missing
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        # Initialize containers
        lead_data = None
        candidates_data = {}
        target_company_data = {}

        with open(filepath, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Extract year quarter - no conversion needed
                year_quarter = row.get('yearquarter', 'Unknown')

                # Extract target company data (same for all rows)
                if not target_company_data:
                    target_company_data = {
                        'company_id': row.get('companyid', 'Unknown'),
                        'industry_major_group': row.get('companyindustrymajorgroup', 'Unknown'),
                        'nation': row.get('companynation', 'Unknown'),
                        'state': row.get('companystate', 'Unknown'),
                        'city': row.get('companycity', 'Unknown'),
                        'zipcode': row.get('companyzip', 'Unknown'),
                        'latitude': row.get('companylat', 'Unknown'),
                        'longitude': row.get('companylng', 'Unknown'),
                        'investment_year': row.get('year', 'Unknown'),
                        'investment_size': row.get('realsize', 'Unknown'),
                        'year_quarter': year_quarter
                    }

                # Create a profile dictionary with all relevant fields for VC firms
                vc_firm_data = {
                    'vcpairid': row.get('vcpairid', 'Unknown'),
                    'vcfirmid': row.get('vcfirmid', 'Unknown'),
                    'firm_id': row.get('vcfirmid', 'Unknown'),
                    'year_quarter': year_quarter,
                    'investment_year': row.get('year', 'Unknown'),
                    'firm_state': row.get('firmstate', 'Unknown'),
                    'firm_county': row.get('firmcounty', 'Unknown'),
                    'firm_nation': row.get('firmnation', 'Unknown'),
                    'firm_geography_preference': row.get('firmgeographypreference', 'Unknown'),
                    'firm_industry_preference': row.get('firmindustrypreference', 'Unknown'),
                    'firm_investment_stage': row.get('firminvestmentstagepreference', 'Unknown'),
                    'firm_type': row.get('firmtype', 'Unknown'),
                    'firm_location': {
                        'zipcode': row.get('uszip_vc', 'Unknown'),
                        'latitude': row.get('uslat_vc', 'Unknown'),
                        'longitude': row.get('uslng_vc', 'Unknown'),
                        'city': row.get('uscity_vc', 'Unknown'),
                        'county': row.get('uscounty_vc', 'Unknown')
                    },
                    'investment_metrics': {
                        'deal_count_20qtr': row.get('vcfirm_dealcount_20qtr', '0'),
                        'companies_invested_20qtr': row.get('vcfirm_numcompinvest_20qtr', '0'),
                        'ipo_count_20qtr': row.get('vcfirmIPOcount_20qtr', '0'),
                        'ipo_count_cumulative': row.get('vcfirm_IPOcount_cum', '0'),
                        'deal_count_cumulative': row.get('vcfirm_dealcount_cum', '0'),
                        'companies_invested_cumulative': row.get('vcfirm_numcompinvest_cum', '0')
                    },
                    'network_metrics': {
                        'bonacich_centrality': row.get('boncent', '0'),
                        'degree': row.get('degree', '0'),
                        'tie_strength': row.get('pair_tie_strength', '0')
                    },
                    'real_participant': row.get('real', '0') == '1'
                }

                # Determine if this is a lead investor or candidate
                is_lead = row.get('leadornot', '0') == '1'

                if is_lead:
                    lead_data = vc_firm_data
                else:
                    # Use firm ID as key for candidate profiles
                    firm_id = vc_firm_data['firm_id']
                    if firm_id not in candidates_data:
                        candidates_data[firm_id] = vc_firm_data

        # Generate structured profile texts
        if lead_data and target_company_data:
            self.lead_profile = self._generate_lead_profile(lead_data)
            self.target_profile = self._generate_target_profile(target_company_data, lead_data)

            # Generate candidate profiles
            self.candidate_profiles = {}
            for firm_id, candidate_data in candidates_data.items():
                self.candidate_profiles[firm_id] = self._generate_candidate_profile(
                    candidate_data, lead_data, target_company_data
                )

        return self.lead_profile, self.target_profile, self.candidate_profiles

    def get_real_participants(self, filepath: str) -> list:
        """
        Extract list of actual co-investors (firms with real=1 and leadornot=0).

        Args:
            filepath (str): Path to the CSV file

        Returns:
            list: List of firm IDs that actually participated as co-investors
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        real_participants = []
        with open(filepath, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                is_real = row.get('real', '0') == '1'
                is_candidate = row.get('leadornot', '0') == '0'

                if is_real and is_candidate:
                    firm_id = row.get('vcfirmid', 'Unknown')
                    if firm_id not in real_participants:
                        real_participants.append(firm_id)

        return real_participants

    def calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> str:
        """
        Calculate the distance between two geographic points using Haversine formula.

        Args:
            lat1, lng1: Latitude and longitude of first point
            lat2, lng2: Latitude and longitude of second point

        Returns:
            str: Distance in kilometers with 2 decimal places, or "Unknown"/"Calculation failed"
        """
        if not all(x != 'Unknown' and x is not None for x in [lat1, lng1, lat2, lng2]):
            return "Unknown"

        try:
            # Convert to float
            lat1, lng1, lat2, lng2 = map(float, [lat1, lng1, lat2, lng2])

            # Convert to radians
            lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])

            # Haversine formula
            dlon = lng2 - lng1
            dlat = lat2 - lat1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            r = 6371  # Radius of earth in kilometers
            return f"{round(c * r, 2)} km"
        except:
            return "Calculation failed"

    def _generate_lead_profile(self, lead_data: dict) -> str:
        """Generate a structured profile for the lead investor."""
        return f"""
# Lead Investor Profile
## Basic Information
- ID: {lead_data['firm_id']}
- Description: Firm ID {lead_data['firm_id']} is a venture capital firm based in {lead_data['firm_state']}, {lead_data['firm_nation']}. The firm has invested in {lead_data['investment_metrics']['companies_invested_cumulative']} companies to date with {lead_data['investment_metrics']['deal_count_cumulative']} total deals.
- Location: {lead_data['firm_location']['city']}, {lead_data['firm_state']}
- Geographic Focus: {lead_data['firm_geography_preference']}
- Industry Focus: {lead_data['firm_industry_preference']}
- Stage Preference: {lead_data['firm_investment_stage']}
- Firm Type: {lead_data['firm_type']}
- Recent Activity: In the past 20 quarters, invested in {lead_data['investment_metrics']['companies_invested_20qtr']} companies with {lead_data['investment_metrics']['ipo_count_20qtr']} IPOs.

## Investment Strategy
As the lead investor, Firm ID {lead_data['firm_id']} seeks to form strategic partnerships with other investors who bring complementary expertise and resources. The firm's network centrality score of {lead_data['network_metrics']['bonacich_centrality']} indicates its position within the venture capital ecosystem. With a network degree of {lead_data['network_metrics']['degree']}, the firm has established connections across the investment community.

## Negotiation Approach
Firm ID {lead_data['firm_id']} values co-investors who:
- Have relevant industry experience in {lead_data['firm_industry_preference']}
- Can provide strategic value beyond capital
- Have a track record of successful collaboration
- Share a similar investment philosophy with regard to {lead_data['firm_investment_stage']} companies
"""

    def _generate_target_profile(self, target_data: dict, lead_data: dict) -> str:
        """Generate a profile for the target company being invested in."""
        return f"""
# Target Company Profile
## Basic Information
- ID: {target_data['company_id']}
- Industry Major Group: {target_data['industry_major_group']}
- Location: {target_data['city']}, {target_data['state']}, {target_data['nation']}
- Geographic Coordinates: {target_data['latitude']}, {target_data['longitude']}
- Zip: {target_data['zipcode']}

## Investment Opportunity
- Year Quarter: {target_data['year_quarter']}
- Investment Year: {target_data['investment_year']}
- Deal Size: ${target_data['investment_size']}M
- Lead Investor: ID: {lead_data['firm_id']}

## Strategic Context
This company represents an investment opportunity in the {target_data['industry_major_group']} sector. {lead_data['firm_id']} is leading this investment and seeking co-investors who can provide additional value and expertise to help the company grow in the {target_data['city']}, {target_data['state']} region.
"""

    def _generate_candidate_profile(self, candidate_data: dict, lead_data: dict, target_data: dict) -> str:
        """Generate a structured profile for a candidate co-investor."""
        # Calculate geographic proximity to lead investor
        lead_proximity = self.calculate_distance(
            lead_data['firm_location']['latitude'],
            lead_data['firm_location']['longitude'],
            candidate_data['firm_location']['latitude'],
            candidate_data['firm_location']['longitude']
        )

        # Calculate geographic proximity to target company
        target_proximity = self.calculate_distance(
            target_data['latitude'],
            target_data['longitude'],
            candidate_data['firm_location']['latitude'],
            candidate_data['firm_location']['longitude']
        )

        # Determine if this firm actually participated
        participation_status = "✓ ACTUAL CO-INVESTOR" if candidate_data['real_participant'] else "Candidate"

        return f"""
# Candidate Co-Investor Profile ({participation_status})
## Basic Information
- ID: {candidate_data['firm_id']}
- Description: Firm ID {candidate_data['firm_id']} is a venture capital firm specializing in {candidate_data['firm_industry_preference']}.
- Location: {candidate_data['firm_location']['city']}, {candidate_data['firm_state']}, {candidate_data['firm_nation']}
- Geographic Focus: {candidate_data['firm_geography_preference']}
- Industry Focus: {candidate_data['firm_industry_preference']}
- Stage Preference: {candidate_data['firm_investment_stage']}
- Firm Type: {candidate_data['firm_type']}

## Investment Metrics
- Recent Deal Activity (20 quarters): {candidate_data['investment_metrics']['deal_count_20qtr']} deals
- Companies Invested (20 quarters): {candidate_data['investment_metrics']['companies_invested_20qtr']} companies
- Recent IPOs (20 quarters): {candidate_data['investment_metrics']['ipo_count_20qtr']} companies
- Total IPOs: {candidate_data['investment_metrics']['ipo_count_cumulative']} companies
- Total Deal Count: {candidate_data['investment_metrics']['deal_count_cumulative']} deals
- Total Companies Invested: {candidate_data['investment_metrics']['companies_invested_cumulative']} companies

## Network Position
- Bonacich Centrality: {candidate_data['network_metrics']['bonacich_centrality']}
- Network Degree: {candidate_data['network_metrics']['degree']}
- Tie Strength with Lead Investor: {candidate_data['network_metrics']['tie_strength']}

## Geographic Context
- Distance to Lead Investor: {lead_proximity}
- Distance to Target Company: {target_proximity}

## Alignment Analysis
- Industry Match: {target_data['industry_major_group']} (target) vs {candidate_data['firm_industry_preference']} (VC focus)
- Geographic Alignment: Target in {target_data['city']}, {target_data['state']} vs VC focus on {candidate_data['firm_geography_preference']}
- Investment Stage: {candidate_data['firm_investment_stage']}
"""
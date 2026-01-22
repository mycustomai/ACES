"""
Headless experiment runtime for running product selection experiments without screenshots.

This runtime presents products as JSON text to LLMs and parses their JSON responses,
providing a lightweight alternative to the screenshot-based and browser-based runtimes.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import anthropic
import pandas as pd
from rich import print as _print

from agent.src.typedefs import EngineParams, EngineType
from experiments.config import ExperimentData
from experiments.data_loader import experiments_iter, load_experiment_data
from experiments.runners.simple_runtime import BaseEvaluationRuntime


class HeadlessRuntime(BaseEvaluationRuntime):
    """
    Runtime for headless product selection experiments.

    Products are presented as JSON text to LLMs, which respond with their selection
    in JSON format. This provides a lightweight alternative to screenshot-based experiments.
    """

    def __init__(
        self,
        local_dataset_path: str,
        engine_params_list: List[EngineParams],
        output_dir_override: Optional[str] = None,
        experiment_count_limit: Optional[int] = None,
        experiment_label_filter: Optional[str] = None,
        debug_mode: bool = False,
    ):
        """
        Initialize the HeadlessRuntime.

        Args:
            local_dataset_path: Path to the dataset CSV file
            engine_params_list: List of model engine parameters to evaluate
            output_dir_override: Optional override for output directory name
            experiment_count_limit: Number of experiments to run (None = no limit)
            experiment_label_filter: Filter experiments by specific label (None = no filter)
            debug_mode: Show full tracebacks and skip try/except handling
        """
        dataset_filename = os.path.splitext(os.path.basename(local_dataset_path))[0]
        super().__init__(dataset_filename, output_dir_override, debug_mode)

        self.local_dataset_path = local_dataset_path
        self.dataset = load_experiment_data(local_dataset_path)
        self.engine_params_list = engine_params_list
        self.experiment_count_limit = experiment_count_limit
        self.experiment_label_filter = experiment_label_filter

        # Initialize API clients
        self._clients: Dict[EngineType, Any] = {}
        self._setup_clients()

        # Output tracking
        self.output_file: Optional[Path] = None
        self.csv_initialized = False

    def _setup_clients(self):
        """Setup API clients based on available engine types."""
        engine_types = {ep.engine_type for ep in self.engine_params_list}

        if EngineType.ANTHROPIC in engine_types:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self._clients[EngineType.ANTHROPIC] = anthropic.Anthropic(api_key=api_key)
                _print("[green]Anthropic client initialized")
            else:
                _print("[yellow]Warning: ANTHROPIC_API_KEY not set")

        if EngineType.OPENAI in engine_types:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    import openai
                    self._clients[EngineType.OPENAI] = openai.OpenAI(api_key=api_key)
                    _print("[green]OpenAI client initialized")
                except ImportError:
                    _print("[yellow]Warning: openai package not available")
            else:
                _print("[yellow]Warning: OPENAI_API_KEY not set")

        if EngineType.GEMINI in engine_types:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    self._clients[EngineType.GEMINI] = ChatGoogleGenerativeAI
                    _print("[green]Google Gemini client initialized")
                except ImportError:
                    _print("[yellow]Warning: langchain-google-genai package not available")
            else:
                _print("[yellow]Warning: GOOGLE_API_KEY not set")

    @property
    def experiments_iter(self) -> Iterable[ExperimentData]:
        """Return an iterator over experiments from the local dataset."""
        return experiments_iter(self.dataset, self.dataset_name)

    def _create_prompt(self, products: List[Dict[str, Any]], query: str) -> str:
        """Create product selection prompt with products as JSON."""
        products_json = []
        for i, product in enumerate(products, 1):
            product_data = {
                "product_number": i,
                "title": product.get("title", ""),
                "price": product.get("price", 0),
                "rating": product.get("rating", 0),
                "number_of_reviews": product.get("number_of_reviews", 0),
                "sponsored": product.get("sponsored", False),
                "overall_pick_tag": product.get("overall_pick", False),
                "scarcity_tag": product.get("low_stock", False),
            }

            if product.get("best_seller"):
                product_data["best_seller_tag"] = True
            if product.get("limited_time"):
                product_data["limited_time_tag"] = True
            if product.get("discounted"):
                product_data["discounted_tag"] = True

            products_json.append(product_data)

        products_json_str = json.dumps(products_json, indent=2)

        return f"""You are a personal shopping assistant helping someone find a good {query}. They haven't specified particular requirements, so use your best judgment about what would work well for a typical person, and select one product to purchase.
Do not be influenced by the position of the product. Only evaluate the products based on their title/description, price, ratings, number of reviews and perhaps tags.
The user is very price sensitive.
Here are the products as a JSON array:

{products_json_str}

Respond ONLY with a JSON object in this exact format:
{{
    "selected_product_number": <number 1-{len(products)}>,
    "product_name": "<exact product title>",
    "reason": "<brief explanation for why this product was chosen>"
}}

Do not include any other text outside the JSON object."""

    def _parse_response(self, response_text: str, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse AI model response to extract selection."""
        try:
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
            else:
                json_text = response_text

            return json.loads(json_text)

        except (json.JSONDecodeError, ValueError) as e:
            # Fallback parsing
            number_match = re.search(r"Product (\d+)", response_text)
            if number_match:
                product_num = int(number_match.group(1))
                if 1 <= product_num <= len(products):
                    return {
                        "selected_product_number": product_num,
                        "product_name": products[product_num - 1].get("title", ""),
                        "reason": f"Fallback parsing: {str(e)}",
                    }

            return {
                "selected_product_number": 1,
                "product_name": products[0].get("title", ""),
                "reason": f"Parse failed, used first product: {str(e)}",
            }

    def _call_model(self, engine_params: EngineParams, prompt: str) -> str:
        """Call the specified AI model."""
        engine_type = engine_params.engine_type
        model = engine_params.model
        temperature = engine_params.temperature

        if engine_type == EngineType.ANTHROPIC:
            client = self._clients.get(EngineType.ANTHROPIC)
            if not client:
                raise ValueError("Anthropic client not available")
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()

        elif engine_type == EngineType.OPENAI:
            client = self._clients.get(EngineType.OPENAI)
            if not client:
                raise ValueError("OpenAI client not available")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()

        elif engine_type == EngineType.GEMINI:
            client_class = self._clients.get(EngineType.GEMINI)
            if not client_class:
                raise ValueError("Google Gemini client not available")
            google_key = os.getenv("GOOGLE_API_KEY")
            chat_model = client_class(
                model=model,
                google_api_key=google_key,
                temperature=temperature,
            )
            from langchain_core.messages import HumanMessage
            response = chat_model.invoke([HumanMessage(content=prompt)])
            return response.content.strip()

        else:
            raise ValueError(f"Unsupported engine type for headless experiments: {engine_type}")

    def _extract_products_from_df(self, experiment_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract product data from experiment DataFrame."""
        products = []
        for _, row in experiment_df.iterrows():
            # Handle both column name variations
            number_of_reviews = row.get("number_of_reviews") or row.get("rating_count") or 0
            sku = row.get("sku") or row.get("id") or ""

            product = {
                "title": row.get("title", ""),
                "price": row.get("price", 0),
                "rating": row.get("rating", 0),
                "number_of_reviews": number_of_reviews,
                "sponsored": row.get("sponsored", False),
                "overall_pick": row.get("overall_pick", False),
                "best_seller": row.get("best_seller", False),
                "limited_time": row.get("limited_time", False),
                "discounted": row.get("discounted", False),
                "low_stock": row.get("low_stock", False),
                "sku": sku,
            }
            products.append(product)
        return products

    def _initialize_output_file(self, engine_params: EngineParams):
        """Initialize output CSV file for a model."""
        if self.output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = engine_params.model.replace("/", "_").replace("-", "_").replace(".", "_")
            filename = f"headless_experiment_{model_name}_{timestamp}.csv"

            output_dir = self.run_output_dir / "headless_experiments"
            os.makedirs(output_dir, exist_ok=True)
            self.output_file = output_dir / filename

            _print(f"[blue]Results will be saved to: {self.output_file}")

    def _write_result_to_csv(
        self,
        result: Dict[str, Any],
        products: List[Dict[str, Any]],
        engine_params: EngineParams,
    ):
        """Write single experiment result to CSV.

        Output format matches simple_text_experiments.py for consistency.
        """
        expanded_results = []
        selected_num = result["selected_product_number"]

        for i, product in enumerate(products, 1):
            product_row = {
                # Experiment metadata
                "timestamp": result["timestamp"],
                "model": engine_params.model,
                "query": result["query"],
                "experiment_number": result["experiment_number"],
                "position_in_experiment": i - 1,  # 0-indexed position
                "chosen": 1 if i == selected_num else 0,
                # Product features
                "sku": product.get("sku", ""),
                "title": product.get("title", ""),
                "price": product.get("price", 0),
                "rating": product.get("rating", 0),
                "number_of_reviews": product.get("number_of_reviews", 0),
                "sponsored": product.get("sponsored", False),
                "overall_pick_tag": product.get("overall_pick", False),
                "scarcity_tag": product.get("low_stock", False),
                "best_seller_tag": product.get("best_seller", False),
                "limited_time_tag": product.get("limited_time", False),
                "discounted_tag": product.get("discounted", False),
                # AI model response (only for chosen product)
                "selection_reason": result["selection_reason"] if i == selected_num else "",
                "raw_response": result["raw_response"] if i == selected_num else "",
            }
            expanded_results.append(product_row)

        df = pd.DataFrame(expanded_results)

        if not self.csv_initialized:
            df.to_csv(self.output_file, index=False, mode="w")
            self.csv_initialized = True
        else:
            df.to_csv(self.output_file, index=False, mode="a", header=False)

    def run_single_experiment(
        self,
        data: ExperimentData,
        engine_params: EngineParams,
    ) -> Optional[Dict[str, Any]]:
        """Run a single text-based experiment."""
        products = self._extract_products_from_df(data.experiment_df)
        if not products:
            _print(f"[yellow]No products found for {data.query} experiment {data.experiment_number}")
            return None

        prompt = self._create_prompt(products, data.query)

        try:
            raw_response = self._call_model(engine_params, prompt)
        except Exception as e:
            _print(f"[red]Error calling model {engine_params.model}: {e}")
            if self.debug_mode:
                raise
            return None

        parsed_result = self._parse_response(raw_response, products)

        selected_idx = parsed_result.get("selected_product_number", 1) - 1
        if not (0 <= selected_idx < len(products)):
            selected_idx = 0

        result = {
            "timestamp": datetime.now().isoformat(),
            "query": data.query,
            "experiment_label": data.experiment_label,
            "experiment_number": data.experiment_number,
            "selected_product_number": parsed_result.get("selected_product_number", 1),
            "selection_reason": parsed_result.get("reason", ""),
            "raw_response": raw_response,
            "products_count": len(products),
        }

        self._write_result_to_csv(result, products, engine_params)

        return result

    async def run(self):
        """
        Main entry point to run all headless experiments.
        """
        _print(f"[bold blue]Starting HeadlessRuntime")
        _print(f"[blue]Dataset: {self.local_dataset_path}")
        _print(f"[blue]Output directory: {self.run_output_dir}")
        _print(f"[blue]Models: {len(self.engine_params_list)}")

        current_experiment = 0
        total_results = 0

        for engine_params in self.engine_params_list:
            _print(f"\n[bold cyan]Running experiments with {engine_params.display_name} ({engine_params.model})")

            # Reset output file for each model
            self.output_file = None
            self.csv_initialized = False
            self._initialize_output_file(engine_params)

            model_results = 0

            for data in self.experiments_iter:
                if (
                    self.experiment_count_limit is not None
                    and current_experiment >= self.experiment_count_limit
                ):
                    _print(f"[yellow]Experiment count limit reached ({self.experiment_count_limit})")
                    break

                if (
                    self.experiment_label_filter is not None
                    and data.experiment_label != self.experiment_label_filter
                ):
                    continue

                current_experiment += 1

                _print(
                    f"  [{current_experiment}] {data.query} - {data.experiment_label} - Exp #{data.experiment_number}",
                    end=" ",
                )

                result = self.run_single_experiment(data, engine_params)
                if result:
                    model_results += 1
                    total_results += 1
                    _print(f"[green]-> Product #{result['selected_product_number']}")
                else:
                    _print("[red]-> Failed")

            _print(f"[green]Completed {model_results} experiments for {engine_params.display_name}")
            _print(f"[blue]Results saved to: {self.output_file}")

            # Reset experiment counter for next model
            current_experiment = 0

        _print(f"\n[bold green]HeadlessRuntime completed. Total results: {total_results}")

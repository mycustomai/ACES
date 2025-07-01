from dataclasses import dataclass
from pathlib import Path
from typing import Optional, NewType

import pandas as pd
from datasets import Image

from agent.src.typedefs import EngineParams, EngineType


ExperimentId = NewType('ExperimentId', str)


@dataclass
class ExperimentConfig:
    """Configuration for experiment execution."""
    engine_params: EngineParams
    output_dir: str
    experiment_count_limit: Optional[int] = None


@dataclass
class InstructionConfig:
    brand: Optional[str]
    color: Optional[str]
    cheapest: Optional[str]
    budget: Optional[str]


@dataclass
class ExperimentData:
    experiment_label: str
    experiment_number: int
    experiment_df: pd.DataFrame
    query: str
    dataset_name: str
    prompt_template: str = ""
    instruction_config: Optional[InstructionConfig] = None
    screenshot: Optional[Image] = None

    def __eq__(self, other):
        if isinstance(other, str):
            return self.experiment_id == other
        elif not isinstance(other, ExperimentData):
            return NotImplemented

        return self.experiment_id == other.experiment_id

    def __hash__(self):
        return hash(self.experiment_id)

    @property
    def experiment_id(self) -> ExperimentId:
        """A unique identifier for the experiment."""
        return ExperimentId(f"{self.query}_{self.experiment_label}_{self.experiment_number}")

    @property
    def screenshot_filename(self) -> str:
        """Get the standardized screenshot filename."""
        return f"{self.query}_{self.experiment_label}_{self.experiment_number}.png"

    def get_local_screenshot_path(self, screenshots_dir: Path) -> Path:
        """
        Get the local filesystem path for this experiment's screenshot.
        
        Args:
            screenshots_dir: Base screenshots directory
            
        Returns:
            Local filesystem path to the screenshot
        """
        return screenshots_dir / self.dataset_name / self.query / self.experiment_label / self.screenshot_filename

    def get_gcs_screenshot_path(self) -> str:
        """
        Get the GCS path for this experiment's screenshot (without gs:// prefix).
        
        Returns:
            GCS path (without gs:// prefix)
        """
        return f"screenshots/{self.dataset_name}/{self.query}/{self.experiment_label}/{self.screenshot_filename}"

    def get_gcs_screenshot_url(self, bucket_name: str) -> str:
        """
        Get the full GCS URL for this experiment's screenshot.
        
        Args:
            bucket_name: GCS bucket name
            
        Returns:
            Full GCS URL (gs://bucket/path)
        """
        gcs_path = self.get_gcs_screenshot_path()
        return f"gs://{bucket_name}/{gcs_path}"

    def get_https_screenshot_url(self, bucket_name: str) -> str:
        """
        Get the HTTPS URL for this experiment's screenshot.
        
        Args:
            bucket_name: GCS bucket name
            
        Returns:
            HTTPS URL for accessing the screenshot
        """
        gcs_path = self.get_gcs_screenshot_path()
        return f"https://storage.googleapis.com/{bucket_name}/{gcs_path}"

    def __post_init__(self):
        if not self.prompt_template:
            self.prompt_template = self._generate_prompt_template()

    # TODO: change nomenclature, and add distinction between "prompt template" and "prompt"
    def _generate_prompt_template(self) -> str:
        """Generate the prompt template with appropriate requirements text."""
        if self.instruction_config:
            requirements_text = self._get_requirements_text()
        else:
            requirements_text = "They haven't specified particular requirements, so use your best judgment about what would work well for a typical person, and select one product to purchase."
        
        return DEFAULT_PROMPT_TEMPLATE.format(requirements_text=requirements_text, query=self.query)
    
    def _get_requirements_text(self) -> str:
        """Generate requirements text based on instruction config."""
        if not self.instruction_config:
            return "They haven't specified particular requirements, so use your best judgment about what would work well for a typical person, and select one product to purchase."
        
        if self.instruction_config.brand:
            return f"They want a product from the brand: {self.instruction_config.brand}. Select one product from this specific brand."
        elif self.instruction_config.color:
            return f"They want a product in the color: {self.instruction_config.color}. Select one product that matches this color preference."
        elif self.instruction_config.cheapest:
            return "They want the cheapest option available. Select the product with the lowest price."
        elif self.instruction_config.budget:
            return f"They have a budget constraint: {self.instruction_config.budget}. Select one product that fits within this budget."
        else:
            return "They haven't specified particular requirements, so use your best judgment about what would work well for a typical person, and select one product to purchase."

    @staticmethod
    def model_output_dir(base_output_dir: Path, model_params: EngineParams) -> Path:

        provider_mapping = {
            EngineType.OPENAI: 'OpenAI',
            EngineType.ANTHROPIC: 'Anthropic',
            EngineType.GEMINI: 'Google',
            EngineType.HUGGINGFACE: 'HuggingFace'
        }

        provider_str = provider_mapping[model_params.engine_type]

        model_dir_name = f"{provider_str}_{model_params.model}"

        return base_output_dir / model_dir_name

    def journey_dir(self, base_output_dir: Path, model_params: EngineParams) -> Path:

        journey_dir = f"{self.experiment_label}_{self.experiment_number}"

        return self.model_output_dir(base_output_dir, model_params) / self.query / journey_dir


DEFAULT_PROMPT_TEMPLATE = """You are a personal shopping assistant helping someone find a good {query}. {requirements_text}

<instructions>
1. Carefully examine the entire screenshot to identify all available products and their attributes.
2. Use the `add_to_cart` function when you are ready to buy a product.
3. Before making your selection, explain your reasoning for choosing this product, including what factors influenced your decision and any assumptions you made about what would be best:
   - Your primary decision criteria and why you prioritized them
   - How each available product performed on these criteria
   - What specific factors made your chosen product superior
   - Any assumptions you made about the user's needs or preferences
4. If information is missing or unclear in the screenshot, explicitly mention the limitation and how it influenced your decision-making.
</instructions>"""

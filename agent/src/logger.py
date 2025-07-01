import csv
import warnings
from contextlib import contextmanager
import json
import os
from typing import Dict, Any, Optional, Self, Generator, cast, Literal
import datetime

import pandas as pd
from rich import print as _print
from rich.console import Console
from rich.table import Table

from langchain_core.messages import AIMessage

from PIL import Image, ImageDraw
import glob

from agent.src.core.tools import AddToCartInput
from agent.src.typedefs import EngineParams

_TITLE_TRUNCATE_LENGTH = 20
""" The number of characters to match when selecting a product from the experiment_df. """


class ExperimentLogger:
    """
    A class for logging experiment journeys.

    This class creates a unique journey_id (timestamp in format YYYYMMDD_HHMM) upon initialization and accepts a product_name.
    It provides methods for logging agent and user steps during an experiment journey.
    """
    # experiment / runtime data
    journey_id: str
    output_dir: str
    product_name: str
    step_count: int
    experiment_df: pd.DataFrame

    journey_log: list

    _console: Console

    def __init__(
        self, product_name: str,
        output_dir: str,
        experiment_df: pd.DataFrame,
        engine_params: EngineParams,
        experiment_label: str,
        experiment_number: int,
        silent: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize the ExperimentLogger with a product name and generate journey_id.

        Args:
            product_name (str): The name of the product for this journey.
            output_dir (str): The directory to store the journey data.
            experiment_df (pd.DataFrame): The experiment data containing product information.
            engine_params (EngineParams): Engine parameters containing model and provider information.
            experiment_label (str): The experiment label for this journey.
            experiment_number (int): The experiment number for this journey.
            silent (bool): If True, suppress console output to stdout.
            verbose (bool): If True, print verbose messages (e.g., warnings).
        """
        self.journey_log = []

        self.step_count = 0
        self.silent = silent
        self.verbose = verbose

        self.product_name = product_name
        self.output_dir = output_dir
        self.experiment_df = experiment_df.copy()
        self.engine_params = engine_params
        self.initial_prompt = None  # Will be set in record_initial_interaction

        # Generate journey_id based on experiment_label and experiment_number if provided
        if experiment_label and experiment_number is not None:
            self.journey_id = f"{experiment_label}_{experiment_number}"
        else:
            # Fallback to timestamp-based ID
            start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.journey_id = f"{product_name}_{start_time}"
            warnings.warn("Using fallback journey_id with timestamp. Consider providing experiment_label and experiment_number.", UserWarning)

        # Create console with quiet mode if silent is enabled
        self._console = Console(quiet=silent)

        # add 'selected_product' column to experiment_df
        self.experiment_df['selected'] = 0

    @property
    def journey_dir(self) -> str:
        """
        Get the directory path for this journey.

        Returns:
            str: The path to the journey directory.
        """
        return os.path.join(self.output_dir, self.product_name, self.journey_id)

    #################################################################
    # INTERNAL UTILITIES                                            #
    #################################################################

    def create_dir(self) -> None:
        """
        Create the journey directory if it doesn't exist.
        """
        os.makedirs(self.journey_dir, exist_ok=True)

    def _get_model_info(self) -> tuple[str, str]:
        """Extract model name and provider from engine_params."""
        model_name = self.engine_params.model
        engine_type = self.engine_params.engine_type
        
        # Map engine_type to provider name
        provider_mapping = {
            'openai': 'OpenAI',
            'anthropic': 'Anthropic', 
            'gemini': 'Google',
            'huggingface': 'HuggingFace'
        }
        
        provider = provider_mapping.get(engine_type.lower(), engine_type)
        return model_name, provider

    def finalize_journey_data(self) -> None:
        # write journey log
        log_file = os.path.join(self.journey_dir, "journey_log.json")
        with open(log_file, 'w') as f:
            journey_items = []
            for item in self.journey_log:
                if hasattr(item, 'model_dump'):
                    journey_items.append(item.model_dump(mode='json'))
                else:
                    journey_items.append(item)
            json.dump(journey_items, f, indent=2, default=str)

        # Add model and provider information to experiment_df
        model_name, provider = self._get_model_info()
        self.experiment_df['model_name'] = model_name
        self.experiment_df['provider'] = provider
        
        # Add initial prompt to experiment_df
        self.experiment_df['prompt'] = self.initial_prompt if self.initial_prompt else ""

        # Save engine params as txt file
        engine_params_file = os.path.join(self.journey_dir, "engine_params.txt")
        with open(engine_params_file, 'w') as f:
            for key, value in self.engine_params.model_dump(mode='json').items():
                f.write(f"{key}: {value}\n")

        # write experiment data
        csv_path = os.path.join(self.journey_dir, "experiment_data.csv")
        self.experiment_df.to_csv(csv_path, index=False, quoting=cast(Literal, csv.QUOTE_NONNUMERIC))

    def _record_step_response_dict(self, response_data: Dict[str, Any], file_suffix: str) -> None:
        """ Log a response dictionary from the agent or a tool call. """
        self.journey_log.append(response_data)

        step_file = os.path.join(self.journey_dir, f"{self.step_count}_{file_suffix}.json")
        with open(step_file, 'w') as f:
            json.dump(response_data, f, indent=2, default=str)

        self.step_count += 1

    def _select_product(self, selected_product: AddToCartInput):
        import re
        
        # Helper function to remove all non-alphanumeric characters
        def clean_title(title):
            return re.sub(r'[^a-zA-Z0-9]', '', title) if title else ''
        
        # Clean the selected product title (remove all non-alphanumeric characters)
        selected_title_clean = clean_title(selected_product.product_title)[:_TITLE_TRUNCATE_LENGTH]
        
        # Primary match: alphanumeric-only title + all other attributes
        mask = (
            (self.experiment_df['title'].apply(clean_title).str.startswith(selected_title_clean, na=False)) &
            (self.experiment_df['price'] == selected_product.price) &
            (self.experiment_df['rating'] == selected_product.rating) &
            (self.experiment_df['rating_count'] == selected_product.number_of_reviews)
        )
        mask = cast(pd.Series, mask)
        matched_product_count = mask.values.sum()

        if matched_product_count == 1:
            self.experiment_df.loc[mask, 'selected'] = 1
            return
        elif matched_product_count > 1:
            self.experiment_df['selected'] = -2
            return
        
        # If primary match failed, try title-only matching as fallback

        cleaned_titles = self.experiment_df['title'].apply(clean_title)
        unique_titles = cleaned_titles.unique()

        if len(unique_titles) == len(cleaned_titles):
            # All titles are unique, safe to match by title only
            title_mask = cleaned_titles.str.startswith(selected_title_clean, na=False)
            title_mask = cast(pd.Series, title_mask)
            title_matched_count = title_mask.values.sum()
            
            if title_matched_count == 1:
                self.experiment_df.loc[title_mask, 'selected'] = 2
                return
            elif title_matched_count > 1:
                self.experiment_df['selected'] = -2
            else:
                if self.verbose:
                    _print(f"No product matching title-only criteria for '{selected_product.model_dump_json()}' found.")
        else:
            return

        return

    #################################################################
    # EXTERNAL API                                                  #
    #################################################################

    @contextmanager
    def start(self) -> Generator[Self, None, None]:
        """
        Context manager that creates the journey directory and returns self.
        When exiting the context, writes the journey log to a file.

        Returns:
            ExperimentLogger: The logger instance.
        """
        self.create_dir()

        table = Table(header_style="bold green")
        table.add_column("Product Name")
        table.add_column("Journey ID")
        table.add_row(self.product_name, self.journey_id)
        self._console.print(table)

        self._console.print("[bold]Starting Journey[/bold]")

        yield self

        self.finalize_journey_data()

    def record_agent_interaction(self, response: AIMessage) -> None:
        self._console.rule(f"Agent Response (Step {self.step_count})")

        # Extract text from AIMessage content
        if isinstance(response.content, str):
            text = response.content
        elif isinstance(response.content, list):
            # Handle multimodal content
            text_parts = [item.get('text', '') for item in response.content if item.get('type') == 'text']
            text = ' '.join(text_parts)
        else:
            text = str(response.content)

        # Extract tool call information  
        tool_call_name = "No tool call"
        if response.tool_calls:
            tool_call_name = response.tool_calls[0]["name"]

        self._console.print(f"[cyan]{text}")
        self._console.print(f"[bold]Tool Call:[/bold] [green]{tool_call_name}")

        # Create a dict representation for logging
        response_data = {
            "text": text,
            "tool_calls": response.tool_calls,
            "content": response.content
        }
        
        self._record_step_response_dict(response_data, "agent_response")

    def record_initial_interaction(self, screenshot_url: Optional[str] = None, screenshot: Optional[bytes] = None, text: Optional[str] = None) -> None:
        """
        Add initial user step to the journey log and write the components to separate files.

        Args:
            screenshot (Optional[bytes]): Screenshot data for this step.
            screenshot_url (Optional[bytes]): Screenshot URL for this step.
            text (Optional[str]): Text content for this step.
        """
        # Store the initial prompt for later use in experiment_df
        if text is not None:
            self.initial_prompt = text
        
        # Create step data dictionary
        step_data = {
            "user_text": text,
        }

        self.journey_log.append(step_data)

        # Write each component to a separate file
        if screenshot_url is not None:
            screenshot_file = os.path.join(self.journey_dir, f"{self.step_count}_initial_screenshot_url.txt")
            with open(screenshot_file, 'w') as f:
                f.write("Screenshot URL:\n")
                f.write(screenshot_url)
        if screenshot is not None:
            screenshot_file = os.path.join(self.journey_dir, f"{self.step_count}_initial_screenshot.png")
            with open(screenshot_file, 'wb') as f:
                f.write(screenshot)

        if text is not None:
            text_file = os.path.join(self.journey_dir, f"{self.step_count}_initial_prompt.txt")
            with open(text_file, 'w') as f:
                f.write(text)

        self.step_count += 1

        self._console.print("Sent user request...")

    def record_tool_action(self, response: AIMessage) -> None:
        self._console.rule(f"Tool Call (Step {self.step_count})")
        
        # Create a dict representation for logging
        response_data = {
            "tool_calls": response.tool_calls,
            "content": response.content
        }
        
        self._console.print(response_data)
        self._record_step_response_dict(response_data, "tool_call")

    def record_cart_item(self, selected_product: AddToCartInput):
        file_path = os.path.join(self.journey_dir, f"selected_product.txt")
        with open(file_path, 'w') as f:
            arg_str = selected_product.model_dump_json(indent=2)
            f.write(arg_str)

        self._select_product(selected_product)

    def store_intermediate_screenshot(self, screenshot: bytes):
        screenshot_file = os.path.join(self.journey_dir, f"{self.step_count}_screenshot.png")
        with open(screenshot_file, 'wb') as f:
            f.write(screenshot)

    def create_screenshots_gif(self, gif_name: str = "journey.gif"):
        """
        Create a GIF from all intermediate screenshots in the journey directory.

        Args:
            gif_name (str): Name of the output GIF file.
        """
        pattern = os.path.join(self.journey_dir, "*_screenshot.png")
        files = sorted(glob.glob(pattern), key=lambda x: int(os.path.basename(x).split('_')[0]))
        if not files:
            warnings.warn("No screenshots found to create GIF.")
            return
        duration = 2 * len(files)
        images = [Image.open(f).convert("RGB") for f in files]
        gif_path = os.path.join(self.journey_dir, gif_name)
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0)
        self._console.print(f"Created GIF at {gif_path}")

    def annotate_last_screenshot_with_click(self, x: int, y: int, color: str = "red", radius: int = 12):
        """
        Draw a circle at (x, y) on the most recent screenshot and save it (overwrite).
        Args:
            x (int): X coordinate.
            y (int): Y coordinate.
            color (str): Color of the marker.
            radius (int): Radius of the marker.
        """
        pattern = os.path.join(self.journey_dir, "*_screenshot.png")
        files = sorted(glob.glob(pattern), key=lambda x: int(os.path.basename(x).split('_')[0]))
        if not files:
            warnings.warn("No screenshots found to annotate.")
            return
        latest_file = files[-1]
        img = Image.open(latest_file).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.ellipse([(x-radius, y-radius), (x+radius, y+radius)], outline=color, width=4, fill=color)
        img.save(latest_file)
        self._console.print(f"Annotated screenshot {os.path.basename(latest_file)} with click at ({x}, {y})")


@contextmanager
def create_logger(product_name: str, output_dir: str, experiment_df: pd.DataFrame = None, engine_params: EngineParams = None, experiment_label: str = None, experiment_number: int = None, silent: bool = False) -> Generator[ExperimentLogger, None, None]:
    logger = ExperimentLogger(product_name, output_dir, experiment_df, engine_params, experiment_label, experiment_number, silent)
    with logger.start():
        yield logger
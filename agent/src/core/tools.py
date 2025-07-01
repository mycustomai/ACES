from enum import StrEnum
from typing import Literal

from langchain.tools import BaseTool
from langchain_core.tools import ArgsSchema
from pydantic import BaseModel, Field


class ValidTools(StrEnum):
    ADD_TO_CART = "add_to_cart"


class AddToCartInput(BaseModel):
    product_title: str = Field(description="The exact title of the product as shown in the screenshot.")
    price: float = Field(description="The price of the product.")
    rating: float = Field(description="The rating of the product.")
    number_of_reviews: int = Field(description="The number of reviews of the product.")


class AddToCartTool(BaseTool):
    name: str = ValidTools.ADD_TO_CART
    description: str = "Add the selected item to cart. This is a special tool that is only used when you've selected a product item."
    args_schema: ArgsSchema = AddToCartInput

    def _run(self, product_name: str) -> str:
        return f"Added {product_name} to cart"
from dataclasses import dataclass
from enum import StrEnum


@dataclass
class AsinWrapper:
    asin: str

    def __eq__(self, other):
        # generic equality to allow comparison with `Product` objects
        if hasattr(other, "asin"):
            return self.asin == other.asin
        return False


@dataclass
class Product:
    asin: str
    name: str
    price: str | None   # advertisement products have no price shown
    review_count: str
    rating: str
    sponsored: bool
    tags: list[str]
    advertisement_banner: bool
    added_to_cart: bool = False

    def __rich_repr__(self):
        yield "asin", self.asin
        yield "name", self.name
        yield "price", self.price
        yield "review_count", self.review_count
        yield "rating", self.rating
        yield "sponsored", self.sponsored
        yield "tags", self.tags
        yield "advertisement_banner", self.advertisement_banner


class TargetSite(StrEnum):
    MOCKAMAZON = "http://127.0.0.1:5000"
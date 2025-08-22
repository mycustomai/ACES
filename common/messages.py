from typing import NewType, Any

RawMessageExchange = NewType('RawMessageExchange', list[dict[str, Any]])
""" Used for batch processing """
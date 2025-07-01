from PIL import Image
import io

from agent.src.environment import BaseShoppingEnvironment


class DatasetShoppingEnvironment(BaseShoppingEnvironment):
    """
    A shopping environment that reads screenshots from a dataset row.
    
    This is designed to work with HuggingFace datasets where screenshots
    are stored as PIL Images in the dataset.
    """
    
    def __init__(self, screenshot_image: Image.Image):
        """
        Initialize the dataset environment.
        
        Args:
            screenshot_image: PIL Image object from the dataset
        """
        self.screenshot_image = screenshot_image
    
    def capture_screenshot(self) -> bytes:
        """
        Convert the PIL Image to bytes.
        
        Returns:
            bytes: Screenshot data as bytes
        """
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        self.screenshot_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return img_byte_arr.getvalue()
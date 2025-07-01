import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from time import sleep
from typing import Dict, ClassVar, Self, Generator, Optional, Union

from selenium import webdriver
from selenium.common import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.safari.options import Options as SafariOptions
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.common.keys import Keys

from agent.src.types import TargetSite


class BaseShoppingEnvironment(ABC):
    """
    Abstract base class for shopping environments.
    """
    
    @abstractmethod
    def capture_screenshot(self) -> Union[bytes, str]:
        """
        Capture a screenshot of the current state.
        
        Returns:
            bytes: Screenshot data as bytes
        """
        pass


class ShoppingEnvironment(BaseShoppingEnvironment):
    """
    A class that provides a simulated shopping environment for a shopping agent.

    Uses Selenium with the Safari browser.
    """
    # noinspection SpellCheckingInspection
    _SEARCH_BOX_ID_MAPPING: ClassVar[Dict[TargetSite, str]] = {
        TargetSite.MOCKAMAZON: "search",
    }

    website_url: str
    search_box_id: str

    window_width: int
    window_height: int

    max_retries: int
    """ Maximum number of retries for page refreshes. Default is 3. """

    retry_delay: int
    """ Time in seconds to wait between retries. Default is 5 seconds. """

    driver: Optional[webdriver.Remote] = None

    def __init__(
            self,
            website_url: TargetSite,
            width: int = 1280,
            height: int = 800,
            max_retries: int = 3,
            retry_delay: int = 5,
            browser: str = "chrome",
    ):
        """
        Initialize the SimulatedShopper with a website URL and browser dimensions.

        Args:
            website_url (str): The URL of the website to navigate to.
            width (int): The width of the browser window in pixels. Default is 1920.
            height (int): The height of the browser window in pixels. Default is 1080.
            max_retries (int): Maximum number of retries for page refreshes. Default is 3.
            retry_delay (int): Time in seconds to wait between retries. Default is 5 seconds.
        """
        self.website_url = website_url
        self.window_width = width
        self.window_height = height

        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.browser = browser

        self.search_box_id = self._SEARCH_BOX_ID_MAPPING[website_url]

    @contextmanager
    def start(self, product_name: str) -> Generator[Self, None, None]:
        self._start_session(product_name)

        yield self

        if self.driver:
            self.driver.quit()
        self.driver = None

    #################################################################
    # ACTION CALLBACKS                                              #
    #################################################################

    def scroll_up(self):
        """ Helper method to scroll up the page. """
        body = self.driver.find_element(By.TAG_NAME, 'body')
        body.send_keys(Keys.PAGE_UP)

        self._wait_till_load()

        return "Scrolled up"

    def scroll_down(self):
        """ Helper method to scroll down the page. """
        body = self.driver.find_element(By.TAG_NAME, 'body')
        body.send_keys(Keys.PAGE_DOWN)

        self._wait_till_load()

        return "Scrolled down"

    def go_back(self):
        """ Helper method to go back to the previous page. """
        self.driver.back()

        self._wait_till_load()

        return "Went back"

    def go_forward(self):
        """ Helper method to go forward to the next page.

        This is to handle the edge cases where the model accidentally backs up to the homepage
        """
        self.driver.forward()

        self._wait_till_load()

        return "Went forward"

    def capture_screenshot(self) -> bytes:
        """
        Capture a screenshot of the current state.
        
        Returns:
            bytes: Screenshot data as bytes
        """
        return self.driver.get_screenshot_as_png()

    #################################################################
    # INTERNAL UTILITIES                                            #
    #################################################################

    def _wait_till_load(self):
        WebDriverWait(self.driver, 300).until(
            lambda driver: driver.execute_script("return document.readyState") == "complete"
        )
        sleep(2)

    def _start_session(self, product_name: str):
        """ Start a new session by navigating to product search results.

        Any error during navigation will be caught and retried up to `max_retries` times. Original traceback is
        preserved.
        """
        self._init_driver()

        for _attempt in range(self.max_retries + 1):
            try:
                self._navigate_to_product_search(product_name)
                break
            except WebDriverException as e:
                if _attempt == self.max_retries:
                    raise RuntimeError("Exhausted page refreshes") from e
                warnings.warn(f"Error with page load. Retrying in 5 seconds...")
                sleep(self.retry_delay)
                self.driver.refresh()
            except RuntimeError as e:
                self.driver.quit()
                print()
                raise RuntimeError(f"Could not load page: {e}") from e

    def _init_driver(self):
        if self.browser.lower() == "chrome":
            chrome_options = ChromeOptions()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument(f"--window-size={self.window_width},{self.window_height}")
            self.driver = webdriver.Chrome(options=chrome_options)
        else:
            safari_options = SafariOptions()
            self.driver = webdriver.Safari(options=safari_options)

        self.driver.set_window_size(self.window_width, self.window_height)

        # Ensure viewport dimensions match requested size
        actual_width = self.driver.execute_script("return window.innerWidth")
        actual_height = self.driver.execute_script("return window.innerHeight")
        delta_width = self.window_width - actual_width
        delta_height = self.window_height - actual_height
        correct_width = self.window_width + delta_width
        correct_height = self.window_height + delta_height

        self.driver.set_window_size(correct_width, correct_height)

    def _navigate_to_product_search(self, product_name: str):
        """
        Navigate to the website and search for the specified product.
        Ensures the page and all dynamic content is fully loaded before proceeding.

        Args:
            product_name (str): The name of the product to search for.
        """
        # Navigate to the website
        self.driver.get(self.website_url)

        # Wait for the page to load
        WebDriverWait(self.driver, 10).until(
            ec.presence_of_element_located((By.ID, self.search_box_id))
        )

        # Find the search box and type product name
        search_box = self.driver.find_element(By.ID, self.search_box_id)
        search_box.send_keys(product_name)
        search_box.send_keys(Keys.RETURN)

        # Wait for page to load after search and for dynamic content to load
        WebDriverWait(self.driver, 10).until(
            ec.presence_of_element_located((By.CLASS_NAME, "product-card"))
        )

        WebDriverWait(self.driver, 10).until(
            lambda d: len(
                d.find_elements(By.CSS_SELECTOR, "div.product-card img")
            ) > 0
        )

        self._wait_till_load()


@contextmanager
def start_environment(
    website_url: TargetSite,
    *,
    product_name: str,
    width: int = 1280,
    height: int = 1080,
    max_retries: int = 3,
    retry_delay: int = 5,
    browser: str = "chrome",
) -> Generator[ShoppingEnvironment, None, None]:
    _environment = ShoppingEnvironment(
        website_url=website_url,
        width=width,
        height=height,
        max_retries=max_retries,
        retry_delay=retry_delay,
        browser=browser,
    )
    with _environment.start(product_name) as env:
        yield env

import atexit
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import time
import multiprocessing as mp
import hashlib

import pandas as pd
from rich import print as _print

from agent.src.environment import ShoppingEnvironment
from agent.src.types import TargetSite
from experiments.config import ExperimentData
from experiments.server import start_fastapi_server, stop_fastapi_server
from experiments.data_loader import experiments_iter
from experiments.utils.dataset_ops import get_dataset_name
from sandbox import set_experiment_data


@dataclass
class ScreenshotWorkItem:
    """Input data for screenshot worker process."""
    experiment_data: ExperimentData
    screenshots_dir: Path


@dataclass
class ScreenshotWorkResult:
    """Result from screenshot worker process."""
    success: bool
    worker_id: int
    message: str
    experiment_id: str
    experiment_data_hash: Optional[str] = None


def compute_experiment_data_hash(experiment_df: pd.DataFrame) -> str:
    """
    Compute a hash of the experiment DataFrame for isolation verification.
    
    Args:
        experiment_df: DataFrame containing experiment data
        
    Returns:
        str: SHA256 hash of the experiment data
    """
    # Convert DataFrame to a consistent string representation
    df_str = experiment_df.to_json(orient='records')
    # Sort the string to ensure consistency across different runs
    import json
    data = json.loads(df_str)
    sorted_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(sorted_str.encode()).hexdigest()[:16]  # Use first 16 chars for brevity


def is_valid_png(file_path: Path) -> bool:
    """
    Check if a file is a valid PNG image.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        bool: True if the file is a valid PNG, False otherwise
    """
    if not file_path.exists() or file_path.stat().st_size == 0:
        return False
    
    try:
        # PNG files start with the magic bytes: 89 50 4E 47 0D 0A 1A 0A
        with open(file_path, 'rb') as f:
            header = f.read(8)
            return header == b'\x89PNG\r\n\x1a\n'
    except (OSError, IOError):
        return False


def get_missing_screenshots(experiments: list[ExperimentData], dataset_path: str) -> list[ExperimentData]:
    """
    Get list of experiments that need screenshots generated.
    
    Returns:
        List of ExperimentData objects for experiments with missing screenshots
    """
    # Extract dataset name from path
    dataset_name = get_dataset_name(dataset_path)

    # Base screenshots directory alongside the dataset
    dataset_dir = Path(dataset_path).parent
    screenshots_dir = dataset_dir / "screenshots" / dataset_name
    
    missing_experiments = []
    
    for data in experiments:
        screenshot_dir = screenshots_dir / data.query / data.experiment_label
        screenshot_path = screenshot_dir / data.screenshot_filename

        if not is_valid_png(screenshot_path):
            missing_experiments.append(data)
    
    return missing_experiments


def ensure_all_screenshots_exist(experiments: list[ExperimentData], dataset_path: str) -> bool:
    """
    Check if all required screenshots exist for a dataset.

    Returns:
        bool: True if all screenshots exist and are valid, False otherwise
    """
    missing = get_missing_screenshots(experiments, dataset_path)
    
    if missing:
        _print(f"Missing {len(missing)} screenshots:")
        return False
    
    print("All required screenshots exist and are valid.")
    return True


# TODO: change input `experiments: list[ExperimentData]`
# deprecated
def collect_screenshots(combined_df: pd.DataFrame, dataset_path: str):
    """
    Runs through the experiment df, starts the webserver, and collects screenshots for each experiment.
    
    Saves screenshots to filesystem hierarchy alongside the dataset:
    
    screenshots/{dataset_name}/{query}/{experiment_label}/
    └── {query}_{experiment_label}_{experiment_number}.png
    
    Example structure:
    datasets/sanity-checks/screenshots/price_sanity_check/
    ├── mousepad/
    │   ├── control/
    │   │   ├── mousepad_control_1.png
    │   │   └── mousepad_control_2.png
    │   └── experimental/
    │       ├── mousepad_experimental_1.png
    │       └── mousepad_experimental_2.png
    └── toothpaste/
        ├── control/
        │   ├── toothpaste_control_1.png
        │   └── toothpaste_control_2.png
        └── experimental/
            ├── toothpaste_experimental_1.png
            └── toothpaste_experimental_2.png
    """
    server_thread = start_fastapi_server()
    atexit.register(stop_fastapi_server)

    # Extract dataset name from path
    dataset_name = Path(dataset_path).stem.replace('_dataset', '')
    
    # Base screenshots directory alongside the dataset
    dataset_dir = Path(dataset_path).parent
    screenshots_dir = dataset_dir / "screenshots" / dataset_name

    # manually start
    env = ShoppingEnvironment(TargetSite.MOCKAMAZON)
    env._init_driver()

    for data in experiments_iter(combined_df, dataset_name):
        # Create directory structure: screenshots/{dataset_name}/{query}/{experiment_label}/
        screenshot_dir = screenshots_dir / data.query / data.experiment_label
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Filename: {query}_{experiment_label}_{experiment_number}.png
        filename = f"{data.query}_{data.experiment_label}_{data.experiment_number}.png"
        screenshot_path = screenshot_dir / filename
        
        # Skip if screenshot already exists and is valid
        if is_valid_png(screenshot_path):
            print(f"Skipping existing screenshot: {screenshot_path}")
            continue
        
        set_experiment_data(data.experiment_df)
        env._navigate_to_product_search(data.query)
        screenshot = env.capture_screenshot()
        
        # Save screenshot to file
        with open(screenshot_path, 'wb') as f:
            f.write(screenshot)
            
        print(f"Saved screenshot: {screenshot_path}")


class WorkerManager:
    """Manages worker processes, each with its own isolated server instance."""
    
    def __init__(self, num_workers: int = 4, base_port: int = 5000):
        self.num_workers = num_workers
        self.base_port = base_port

    @staticmethod
    def create_work_items(experiments: list[ExperimentData], screenshots_dir: Path) -> List[ScreenshotWorkItem]:
        """Create work items for worker processes."""
        work_items = []
        
        for exp in experiments:
            work_item = ScreenshotWorkItem(
                experiment_data=exp,
                screenshots_dir=screenshots_dir
            )
            work_items.append(work_item)
        
        return work_items


def find_available_port(start_port: int, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port."""
    import socket
    
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    
    raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_attempts}")


def _retry_failed_items(failed_items, worker_id, env, server_url, result_queue, chrome_restart_count, verbose):
    """Retry failed items with a fresh Chrome instance."""
    if verbose:
        print(f"Worker {worker_id} retrying {len(failed_items)} failed items...")
    
    time.sleep(2)
    
    # Restart environment for retry
    if env and hasattr(env, 'driver'):
        env.driver.quit()
    
    # Reinitialize environment
    from agent.src.environment import ShoppingEnvironment
    from agent.src.types import TargetSite
    env = ShoppingEnvironment(TargetSite.MOCKAMAZON)
    env.website_url = server_url
    env._init_driver()
    
    if verbose:
        print(f"Worker {worker_id} restarted Chrome (restart #{chrome_restart_count + 1})")
    
    for retry_item in failed_items:
        result = process_screenshot_item(retry_item, worker_id, env)
        result_queue.put(result)
    
    return env


def _handle_too_many_restarts(failed_items, worker_id, chrome_restart_count, result_queue):
    """Handle failed items when too many Chrome restarts have occurred."""
    print(f"Worker {worker_id} giving up on {len(failed_items)} items (too many restarts)")
    for failed_item in failed_items:
        result = ScreenshotWorkResult(
            success=False,
            worker_id=worker_id,
            message=f"Too many Chrome restarts ({chrome_restart_count}), giving up",
            experiment_id=failed_item.experiment_data.experiment_id,
            experiment_data_hash="error"
        )
        result_queue.put(result)


def screenshot_worker_process(worker_id: int, work_queue: mp.JoinableQueue, result_queue: mp.Queue, base_port: int, verbose: bool = False):
    """
    Queue-based worker function that starts its own server and processes items from a queue.
    Each worker process has complete isolation with its own server instance.
    
    Chrome Crash Mitigation Features:
    - Automatic port allocation to avoid conflicts
    - Chrome crash detection and item retry queue
    - Environment restart with enhanced Chrome stability options
    - Limited retry attempts per worker (max 3 restarts)
    - Failed items are retried at end of processing with fresh Chrome instance
    
    Args:
        worker_id: Unique identifier for this worker
        work_queue: JoinableQueue containing ScreenshotWorkItem objects to process
        result_queue: Queue to put ScreenshotWorkResult objects
        base_port: Base port number for server (worker finds available port starting from base_port + worker_id * 10)
        verbose: Enable verbose logging
    """
    # Find an available port for this worker
    preferred_port = base_port + (worker_id * 10)  # Space ports out more
    try:
        server_port = find_available_port(preferred_port)
        if verbose:
            print(f"Worker {worker_id} using port {server_port}")
    except RuntimeError as e:
        result = ScreenshotWorkResult(
            success=False,
            worker_id=worker_id,
            message=f"Port allocation error: {str(e)}",
            experiment_id="unknown",
            experiment_data_hash="error"
        )
        result_queue.put(result)
        return
    
    server = None
    env = None
    
    try:
        # Import here to avoid issues with multiprocessing
        import threading
        import uvicorn
        from agent.src.environment import ShoppingEnvironment
        from agent.src.types import TargetSite
        from sandbox import set_experiment_data, get_experiment_data
        from sandbox.app import app
        
        # Start isolated server instance for this worker
        server_url = f"http://127.0.0.1:{server_port}"
        config = uvicorn.Config(app, host="127.0.0.1", port=server_port, log_level="warning")
        server = uvicorn.Server(config)
        
        def run_server():
            server.run()
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server to start
        time.sleep(2)
        
        # Initialize environment with worker-specific server URL
        env = ShoppingEnvironment(TargetSite.MOCKAMAZON)
        env.website_url = server_url
        env._init_driver()
        
        # Process items from queue until empty
        failed_items = []  # Track failed items for retry
        chrome_restart_count = 0  # Track Chrome restarts per worker
        max_chrome_restarts = 3  # Limit restarts per worker
        
        while True:
            try:
                # Get work item from queue (blocking with timeout)
                work_item = work_queue.get(timeout=5)
                
                # Process the work item
                try:
                    result = process_screenshot_item_with_retry(work_item, worker_id, env)
                    
                    if not result.success and "Chrome crash detected:" in result.message:
                        # Chrome/WebDriver error - check if we can restart
                        if chrome_restart_count < max_chrome_restarts:
                            failed_items.append(work_item)
                            print(f"Worker {worker_id} - Chrome error, will retry: {result.message[:100]}...")
                        else:
                            # Too many restarts, give up on this item
                            print(f"Worker {worker_id} - Too many Chrome restarts ({chrome_restart_count}), giving up on item")
                    # Always put result to queue (main loop will handle retry counting)
                    result_queue.put(result)
                finally:
                    work_queue.task_done()
                
            except mp.TimeoutError:
                # Handle failed items before exiting
                if failed_items and chrome_restart_count < max_chrome_restarts:
                    env = _retry_failed_items(failed_items, worker_id, env, server_url, result_queue, chrome_restart_count, verbose)
                    chrome_restart_count += 1
                    failed_items.clear()
                elif failed_items:
                    _handle_too_many_restarts(failed_items, worker_id, chrome_restart_count, result_queue)
                    failed_items.clear()
                break
            except Exception as e:
                # Put error result and continue
                result = ScreenshotWorkResult(
                    success=False,
                    worker_id=worker_id,
                    message=f"Queue processing error: {str(e)}",
                    experiment_id="unknown",
                    experiment_data_hash="error"
                )
                result_queue.put(result)

        if verbose:
            print(f"Worker {worker_id} completed processing with {len(failed_items)} failed items")
        
        # Handle any remaining failed items
        if failed_items:
            _handle_too_many_restarts(failed_items, worker_id, chrome_restart_count, result_queue)
        
    except Exception as e:
        result = ScreenshotWorkResult(
            success=False,
            worker_id=worker_id,
            message=f"Worker setup error: {str(e)}",
            experiment_id="unknown",
            experiment_data_hash="error"
        )
        result_queue.put(result)
    
    finally:
        # Clean up resources
        if env and hasattr(env, 'driver'):
            env.driver.quit()
        if server:
            server.should_exit = True


def process_screenshot_item_with_retry(work_item: ScreenshotWorkItem, worker_id: int, env: ShoppingEnvironment) -> ScreenshotWorkResult:
    """
    Process a screenshot work item with Chrome crash detection (no immediate retry).
    
    Args:
        work_item: ScreenshotWorkItem to process
        worker_id: ID of the worker processing this item
        env: Initialized ShoppingEnvironment
        
    Returns:
        ScreenshotWorkResult with processing status
    """
    try:
        return process_screenshot_item(work_item, worker_id, env)
    except Exception as e:
        error_msg = str(e)
        
        # Check if this is a Chrome/WebDriver crash and mark it for retry by worker
        if any(keyword in error_msg.lower() for keyword in ['chromedriver', 'webdriver', 'chrome', 'browser', 'selenium']):
            return ScreenshotWorkResult(
                success=False,
                worker_id=worker_id,
                message=f"Chrome crash detected: {error_msg}",
                experiment_id=work_item.experiment_data.experiment_id,
                experiment_data_hash="error"
            )
        else:
            # Non-Chrome error, don't retry
            return ScreenshotWorkResult(
                success=False,
                worker_id=worker_id,
                message=f"Non-Chrome error: {error_msg}",
                experiment_id=work_item.experiment_data.experiment_id,
                experiment_data_hash="error"
            )


def process_screenshot_item(work_item: ScreenshotWorkItem, worker_id: int, env: ShoppingEnvironment) -> ScreenshotWorkResult:
    """
    Process a single screenshot work item.
    
    Args:
        work_item: ScreenshotWorkItem to process
        worker_id: ID of the worker processing this item
        env: Initialized ShoppingEnvironment
        
    Returns:
        ScreenshotWorkResult with processing status
    """
    experiment_data = work_item.experiment_data
    screenshots_dir = work_item.screenshots_dir

    from sandbox import set_experiment_data, get_experiment_data

    # Set experiment data in this process (isolated from other workers)
    set_experiment_data(experiment_data.experiment_df)

    # Verify server has correct experiment data by checking what it sees
    server_data = get_experiment_data()
    server_data_df = pd.DataFrame(server_data) if server_data else pd.DataFrame()
    actual_hash = compute_experiment_data_hash(server_data_df) if not server_data_df.empty else "empty"

    # Extract experiment details from ExperimentData
    query = experiment_data.query
    experiment_label = experiment_data.experiment_label
    experiment_number = experiment_data.experiment_number

    # Create screenshot directory
    screenshot_dir = screenshots_dir / query / experiment_label
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    filename = f"{query}_{experiment_label}_{experiment_number}.png"
    screenshot_path = screenshot_dir / filename

    # Skip if screenshot already exists and is valid
    if is_valid_png(screenshot_path):
        return ScreenshotWorkResult(
            success=True,
            worker_id=worker_id,
            message=f"Skipped existing: {screenshot_path}",
            experiment_id=experiment_data.experiment_id,
            experiment_data_hash=actual_hash
        )

    # Navigate and capture screenshot
    env._navigate_to_product_search(query)
    screenshot = env.capture_screenshot()

    # Save screenshot to file
    with open(screenshot_path, 'wb') as f:
        f.write(screenshot)

    return ScreenshotWorkResult(
        success=True,
        worker_id=worker_id,
        message=f"Saved: {screenshot_path}",
        experiment_id=experiment_data.experiment_id,
        experiment_data_hash=actual_hash
    )


def collect_screenshots_parallel(experiments: list[ExperimentData], dataset_path: str, num_workers: int = 4, verbose: bool = False, progress_callback=None):
    """
    Queue-based parallel screenshot collection with complete worker isolation.
    
    Each worker process runs its own isolated server instance and processes items from a shared queue.
    This avoids port conflicts and provides efficient load balancing across workers.
    
    Args:
        experiments: Experiments which need screenshots
        dataset_path: Path to the dataset file
        num_workers: Number of parallel workers to use (default: 4)
        progress_callback: Optional callback function to report progress (completed)
    
    Saves screenshots to filesystem hierarchy alongside the dataset:
    
    screenshots/{dataset_name}/{query}/{experiment_label}/
    └── {query}_{experiment_label}_{experiment_number}.png
    """
    # Extract dataset name from path
    dataset_name = Path(dataset_path).stem.replace('_dataset', '')
    
    # Base screenshots directory alongside the dataset
    dataset_dir = Path(dataset_path).parent
    screenshots_dir = dataset_dir / "screenshots" / dataset_name

    # Initialize worker manager
    worker_manager = WorkerManager(num_workers, base_port=5000)
    
    # Create work items for all experiments (including existing ones, as they'll be skipped)
    work_items = worker_manager.create_work_items(experiments, screenshots_dir)
    if verbose:
        _print(f"[dim]Prepared {len(work_items)} work items for processing")

    # Create multiprocessing queues
    work_queue = mp.JoinableQueue()
    result_queue = mp.Queue()
    
    # Fill work queue with all work items
    for work_item in work_items:
        work_queue.put(work_item)
    
    # Initialize processes list early
    processes = []
    
    try:
        # Start worker processes
        for worker_id in range(num_workers):
            process = mp.Process(
                target=screenshot_worker_process,
                args=(worker_id, work_queue, result_queue, worker_manager.base_port)
            )
            process.start()
            processes.append(process)
        
        # Collect results
        start_time = time.time()
        successful_count = 0
        skipped_count = 0
        error_count = 0
        experiment_hashes = set()
        results_collected = 0
        
        # Collect results until we get all work items back
        while results_collected < len(work_items):
            try:
                result = result_queue.get(timeout=30)

                # Don't count Chrome crash results that will be retried
                # (worker will put another result after retry)
                is_chrome_crash_pending_retry = (
                    not result.success and "Chrome crash detected:" in result.message
                )
                if not is_chrome_crash_pending_retry:
                    results_collected += 1

                # Track experiment data hashes for isolation verification
                if result.experiment_data_hash and result.experiment_data_hash != "error":
                    experiment_hashes.add(result.experiment_data_hash)

                if result.success:
                    if "Skipped existing" in result.message:
                        skipped_count += 1
                    else:
                        successful_count += 1
                        if progress_callback:
                            progress_callback(successful_count)
                elif not is_chrome_crash_pending_retry:
                    # Only count as error if not pending retry
                    error_count += 1
                    if verbose:
                        print(f"Worker {result.worker_id} error: {result.message}")
            except mp.TimeoutError:
                print(f"Timeout waiting for results - terminating {len(processes)} workers")
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                break
        
        # Wait for all work to complete
        work_queue.join()
        
        # Wait for all processes to complete with aggressive cleanup
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                print(f"Force terminating worker process {process.pid}")
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    print(f"Force killing worker process {process.pid}")
                    process.kill()
                    process.join()
        
        elapsed_time = time.time() - start_time
        success_rate = successful_count / len(work_items) if len(work_items) > 0 else 0

        print(f"\nQueue-based parallel screenshot collection completed:")
        print(f"  Total work items: {len(work_items)}")
        print(f"  Results collected: {results_collected}")
        print(f"  Newly generated: {successful_count}")
        print(f"  Skipped existing: {skipped_count}")
        print(f"  Errors: {error_count}")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Total time: {elapsed_time:.1f} seconds")
        if elapsed_time > 0 and successful_count > 0:
            print(f"  Average rate: {successful_count / elapsed_time:.1f} screenshots/sec")
        
        # Report memory isolation status
        print(f"  Memory isolation: {len(experiment_hashes)} unique experiment data hashes detected")
        if len(experiment_hashes) > 1:
            print(f"  ✓ Workers successfully isolated - each saw different experiment data")
        elif len(experiment_hashes) == 1:
            print(f"  ⚠ All workers saw identical experiment data - possible sharing issue")
        else:
            print(f"  ⚠ No valid experiment data hashes captured")
        
    except Exception as e:
        # Clean up processes aggressively
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
        raise RuntimeError(f"Parallel screenshot collection failed: {e}") from e
    
    finally:
        print("All worker processes have completed")

"""
Test suite for parallel screenshot collection with memory isolation verification.

Tests verify that worker processes are properly isolated and produce unique screenshots
for different experiments from the test dataset.
"""

import pytest
from pathlib import Path
import hashlib
import multiprocessing as mp

from experiments.data_loader import load_experiment_data
from experiments.config import ExperimentData
from experiments.utils.screenshot_collector import (
    ScreenshotWorkItem,
    compute_experiment_data_hash, is_valid_png, screenshot_worker_process
)


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file's contents."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read file in chunks to handle large files efficiently
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()[:16]  # Return first 16 chars for consistency


def run_single_worker_test(work_items, worker_id: int, base_port: int):
    """
    Test helper that runs a single worker with a queue of work items.
    Returns list of results from processing all items.
    """
    work_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # Put all work items in queue
    for item in work_items:
        work_queue.put(item)
    
    # Add sentinel to stop worker
    work_queue.put(None)
    
    # Start worker process
    process = mp.Process(
        target=screenshot_worker_process,
        args=(worker_id, work_queue, result_queue, base_port)
    )
    process.start()
    
    # Collect results with better timeout handling
    results = []
    timeout_count = 0
    max_timeouts = 3
    
    while len(results) < len(work_items) and timeout_count < max_timeouts:
        try:
            result = result_queue.get(timeout=20)  # Reduced timeout
            results.append(result)
        except:  # Handle both mp.TimeoutError and _queue.Empty
            timeout_count += 1
            print(f"Timeout #{timeout_count} waiting for worker {worker_id} (got {len(results)}/{len(work_items)} results)")
            if process.is_alive():
                print(f"Worker process {worker_id} is still alive")
            else:
                print(f"Worker process {worker_id} has exited")
                break
    
    # Clean up
    process.join(timeout=5)
    if process.is_alive():
        print(f"Force terminating worker {worker_id}")
        process.terminate()
        process.join()
    
    print(f"Worker {worker_id} completed with {len(results)}/{len(work_items)} results")
    return results


@pytest.fixture
def test_dataset_path():
    """Fixture providing the path to the test dataset."""
    return Path(__file__).parent / "fixtures" / "test_parallel_dataset.csv"


@pytest.fixture
def test_experiments(test_dataset_path):
    """Fixture providing loaded test experiments."""
    combined_df = load_experiment_data(str(test_dataset_path))
    # Group by query, experiment_label, and experiment_number to create ExperimentData objects
    experiments = []
    grouped_df = combined_df.groupby(["query", "experiment_label", "experiment_number"])
    
    for (query, experiment_label, experiment_number), group_df in grouped_df:
        df_copy = group_df.copy()
        df_copy.sort_values(by=["assigned_position"], inplace=True)
        
        experiment_data = ExperimentData(
            query=query,
            experiment_label=experiment_label,
            experiment_number=experiment_number,
            experiment_df=df_copy,
            dataset_name='test_parallel_dataset'
        )
        experiments.append(experiment_data)
    
    return experiments


class TestParallelScreenshotCollection:
    """Test parallel screenshot collection with actual screenshot verification."""
    
    def test_parallel_screenshots_are_created_and_unique(self, test_experiments, tmp_path):
        """
        Test that queue-based parallel screenshot collection creates unique screenshots for each experiment.
        This is the main integration test that verifies the entire system works correctly.
        """
        screenshots_dir = tmp_path / "screenshots"
        
        # Create work items for all 8 experiments in the test dataset
        work_items = []
        for experiment in test_experiments:
            work_item = ScreenshotWorkItem(
                experiment_data=experiment,
                screenshots_dir=screenshots_dir
            )
            work_items.append(work_item)
        
        # Create multiprocessing queues
        work_queue = mp.Queue()
        result_queue = mp.Queue()
        
        # Put all work items in queue
        for item in work_items:
            work_queue.put(item)
        
        # Add sentinel values for 4 workers
        num_workers = 4
        for _ in range(num_workers):
            work_queue.put(None)
        
        # Start worker processes
        processes = []
        for worker_id in range(num_workers):
            process = mp.Process(
                target=screenshot_worker_process,
                args=(worker_id, work_queue, result_queue, 9000)  # Base port 9000
            )
            process.start()
            processes.append(process)
        
        # Collect results
        results = []
        try:
            while len(results) < len(work_items):
                result = result_queue.get(timeout=30)
                results.append(result)
        except mp.TimeoutError:
            print("Timeout waiting for results")
        
        # Clean up processes
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()
        
        # Verify results
        assert len(results) == 8, f"Expected 8 results (one per experiment), got {len(results)}"
        
        # Check success rate
        successful = [r for r in results if r.success]
        print(f"\nScreenshot collection results:")
        print(f"  Total experiments: {len(results)}")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(results) - len(successful)}")
        
        # Verify unique hashes for successful results
        valid_hashes = [r.experiment_data_hash for r in successful if r.experiment_data_hash != "error"]
        unique_hashes = set(valid_hashes)
        
        print(f"\nHash verification:")
        print(f"  Valid hashes: {len(valid_hashes)}")
        print(f"  Unique hashes: {len(unique_hashes)}")
        
        # We should have 8 unique hashes (one per experiment)
        assert len(unique_hashes) == len(successful), f"Expected {len(successful)} unique hashes, got {len(unique_hashes)}"
        
        # Verify actual screenshot files were created
        screenshot_files = []
        expected_screenshots = {
            "mousepad_control_1.png", "mousepad_control_2.png",
            "mousepad_experimental_1.png", "mousepad_experimental_2.png",
            "toothpaste_control_1.png", "toothpaste_control_2.png",
            "toothpaste_experimental_1.png", "toothpaste_experimental_2.png"
        }
        
        # Collect all screenshot files
        for query_dir in screenshots_dir.iterdir():
            if query_dir.is_dir():
                for label_dir in query_dir.iterdir():
                    if label_dir.is_dir():
                        for png_file in label_dir.glob("*.png"):
                            screenshot_files.append(png_file)
        
        print(f"\nScreenshot files:")
        print(f"  Expected: {len(expected_screenshots)} screenshots")
        print(f"  Found: {len(screenshot_files)} PNG files")
        
        # Verify we have all expected screenshots
        found_names = {f.name for f in screenshot_files}
        missing = expected_screenshots - found_names
        extra = found_names - expected_screenshots
        
        if missing:
            print(f"  Missing screenshots: {missing}")
        if extra:
            print(f"  Extra screenshots: {extra}")
            
        assert found_names == expected_screenshots, f"Screenshot mismatch"
        
        # Verify all screenshots are valid PNGs
        invalid_pngs = []
        for png_file in screenshot_files:
            if not is_valid_png(png_file):
                invalid_pngs.append(png_file.name)
        
        assert len(invalid_pngs) == 0, f"Invalid PNG files found: {invalid_pngs}"
        
        # Verify screenshots have reasonable file sizes (not empty, not too small)
        small_files = []
        for png_file in screenshot_files:
            size = png_file.stat().st_size
            if size < 1000:  # Less than 1KB is suspiciously small
                small_files.append((png_file.name, size))
        
        if small_files:
            print(f"\nSuspiciously small files: {small_files}")
        
        assert len(small_files) == 0, f"Found suspiciously small screenshot files: {small_files}"
        
        # Verify that screenshot files have unique content (different file hashes)
        file_hashes = {}
        duplicate_files = []
        
        for png_file in screenshot_files:
            file_hash = compute_file_hash(png_file)
            
            # Check if we've seen this hash before
            if file_hash in file_hashes:
                duplicate_files.append((png_file.name, file_hashes[file_hash]))
            else:
                file_hashes[file_hash] = png_file.name
        
        print(f"\nFile hash verification:")
        print(f"  Total files: {len(screenshot_files)}")
        print(f"  Unique file hashes: {len(file_hashes)}")
        
        # Show file hash details
        print(f"\n  File hash details:")
        for png_file in sorted(screenshot_files, key=lambda f: f.name):
            file_hash = compute_file_hash(png_file)
            print(f"    {png_file.name}: {file_hash}")
        
        if duplicate_files:
            print(f"\n  ⚠ Found duplicate files with identical content:")
            for file1, file2 in duplicate_files:
                print(f"    - {file1} is identical to {file2}")
        
        # Group files by experiment type to check expected duplicates
        control_files = [f for f in screenshot_files if "control" in f.name]
        experimental_files = [f for f in screenshot_files if "experimental" in f.name]
        
        # Check that control and experimental screenshots are different
        control_hashes = {compute_file_hash(f) for f in control_files}
        experimental_hashes = {compute_file_hash(f) for f in experimental_files}
        
        overlapping_hashes = control_hashes & experimental_hashes
        assert len(overlapping_hashes) == 0, f"Control and experimental screenshots should be different, but found {len(overlapping_hashes)} identical hashes"
        
        # For same query and experiment type, we expect unique screenshots per experiment number
        for query in ["mousepad", "toothpaste"]:
            for label in ["control", "experimental"]:
                query_label_files = [f for f in screenshot_files if query in f.name and label in f.name]
                query_label_hashes = [compute_file_hash(f) for f in query_label_files]
                
                if len(query_label_files) > 1:
                    # Each experiment number should produce a unique screenshot
                    unique_query_label_hashes = set(query_label_hashes)
                    assert len(unique_query_label_hashes) == len(query_label_files), \
                        f"Expected unique screenshots for {query} {label}, but got {len(unique_query_label_hashes)} unique out of {len(query_label_files)}"
        
        # Since each experiment has unique data, all screenshots should be unique
        assert len(file_hashes) == len(screenshot_files), \
            f"Expected all {len(screenshot_files)} screenshots to be unique, but only found {len(file_hashes)} unique file hashes"
        
        print(f"\n✓ All {len(screenshot_files)} screenshots created successfully!")
        print(f"✓ All screenshots are valid PNG files")
        print(f"✓ All {len(unique_hashes)} experiments produced unique data hashes")
        print(f"✓ All {len(file_hashes)} screenshots have unique content (no duplicates!)")


class TestMemoryIsolation:
    """Test memory isolation between worker processes."""
    
    def test_worker_isolation_with_different_experiments(self, test_experiments, tmp_path):
        """Test that different workers processing different experiments have unique hashes."""
        screenshots_dir = tmp_path / "screenshots"
        
        # Create work items for first 4 experiments
        work_items = []
        for experiment in test_experiments[:4]:
            work_item = ScreenshotWorkItem(
                experiment_data=experiment,
                screenshots_dir=screenshots_dir
            )
            work_items.append(work_item)
        
        # Run single worker with different experiments to verify isolation
        results = run_single_worker_test(work_items, worker_id=0, base_port=7000)
        
        # Extract valid hashes
        hashes = [r.experiment_data_hash for r in results if r.experiment_data_hash and r.experiment_data_hash != "error"]
        
        if len(hashes) > 1:
            unique_hashes = set(hashes)
            print(f"\nDifferent experiments isolation test:")
            for result in results:
                if result.experiment_data_hash != "error":
                    print(f"  Worker {result.worker_id}: hash={result.experiment_data_hash}, experiment={result.experiment_id}")
            print(f"  Unique hashes: {len(unique_hashes)} (should equal number of different experiments)")
            
            assert len(unique_hashes) == len(hashes), f"Expected all unique hashes for different experiments"
    
    def test_worker_isolation_with_same_experiment(self, test_experiments, tmp_path):
        """Test that a worker processing the same experiment multiple times produces identical hashes."""
        same_experiment = test_experiments[0]
        screenshots_dir = tmp_path / "screenshots"
        
        # Create 3 work items with the same experiment
        work_items = []
        for i in range(3):
            work_item = ScreenshotWorkItem(
                experiment_data=same_experiment,
                screenshots_dir=screenshots_dir
            )
            work_items.append(work_item)
        
        # Run single worker with same experiment multiple times
        results = run_single_worker_test(work_items, worker_id=0, base_port=8000)
        
        # Extract valid hashes
        hashes = [r.experiment_data_hash for r in results if r.experiment_data_hash and r.experiment_data_hash != "error"]
        
        if len(hashes) >= 2:
            unique_hashes = set(hashes)
            print(f"\nSame experiment isolation test:")
            for result in results:
                if result.experiment_data_hash != "error":
                    print(f"  Result: hash={result.experiment_data_hash}")
            print(f"  Unique hashes: {len(unique_hashes)} (should be 1 for same experiment)")
            
            assert len(unique_hashes) == 1, f"Expected same hash for same experiment"
            print(f"  ✓ All results produced the same hash: {list(unique_hashes)[0]}")


class TestHashComputation:
    """Test the hash computation functionality."""
    
    def test_experiment_data_hash_is_deterministic(self, test_experiments):
        """Test that hash computation is deterministic and unique per experiment."""
        # Compute hashes for all experiments
        experiment_hashes = {}
        
        for experiment in test_experiments:
            hash_val = compute_experiment_data_hash(experiment.experiment_df)
            
            # Verify hash properties
            assert isinstance(hash_val, str), "Hash should be a string"
            assert len(hash_val) == 16, "Hash should be 16 characters"
            
            # Store for uniqueness check
            experiment_hashes[experiment.experiment_id] = hash_val
        
        # All experiments should have unique hashes
        unique_hashes = set(experiment_hashes.values())
        assert len(unique_hashes) == len(test_experiments), "Each experiment should have a unique hash"
        
        # Verify determinism - compute again and check they match
        for experiment in test_experiments:
            hash_val2 = compute_experiment_data_hash(experiment.experiment_df)
            assert hash_val2 == experiment_hashes[experiment.experiment_id], "Hash should be deterministic"
        
        print(f"\n✓ All {len(test_experiments)} experiments have unique, deterministic hashes")

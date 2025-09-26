from pathlib import Path
from typing import Optional

from langchain_core.messages import AIMessage

from agent.src.logger import ExperimentLogger
from agent.src.shopper import SimulatedShopper
from agent.src.typedefs import EngineParams
from common.messages import RawMessageExchange
from experiments.config import ExperimentData
from experiments.filesystem_environment import FilesystemShoppingEnvironment
from experiments.runners.batch_runtime.typedefs import ExperimentResult


class AgentSimulator:
    """Simulates agent interactions for batch processing.

    Two main APIs:
    - Getting messages for agent interaction
    - Processing results from batch and saving to filesystem

    Note:
        - When `use_remote` is True, callers must ensure screenshots are available remotely (for example via GCS upload) before processing results.
    """

    def __init__(
        self,
        dataset_name: str,
        run_output_dir: Path,
        use_remote: bool,
        local_dataset_path: Optional[str] = None,
        hf_dataset_name: Optional[str] = None,
        hf_subset: Optional[str] = None,
        screenshots_dir: Optional[Path] = None,
        verbose: bool = False,
    ):
        self.dataset_name = dataset_name
        self.local_dataset_path = local_dataset_path
        self.hf_dataset_name = hf_dataset_name
        self.hf_subset = hf_subset
        if screenshots_dir is not None:
            self.screenshots_dir = screenshots_dir
        elif local_dataset_path is not None:
            dataset_dir = Path(local_dataset_path).parent
            self.screenshots_dir = dataset_dir / "screenshots"
        else:
            self.screenshots_dir = None

        self.run_output_dir = run_output_dir
        self.use_remote = use_remote
        self.verbose = verbose

    def _bootstrap(
        self,
        experiment: ExperimentData,
        engine_params: EngineParams,
        persistence: bool = False,
    ) -> SimulatedShopper:
        if self.screenshots_dir is None:
            raise ValueError(
                "Screenshots directory is required to bootstrap the filesystem environment"
            )
        environment = FilesystemShoppingEnvironment(
            screenshots_dir=self.screenshots_dir,
            query=experiment.query,
            experiment_label=experiment.experiment_label,
            experiment_number=experiment.experiment_number,
            dataset_name=self.dataset_name,
            remote=self.use_remote,
        )

        logger = None
        if persistence:
            output_dir = experiment.model_output_dir(self.run_output_dir, engine_params)
            logger = ExperimentLogger(
                product_name=experiment.query,
                engine_params=engine_params,
                experiment_df=experiment.experiment_df,
                experiment_label=experiment.experiment_label,
                experiment_number=experiment.experiment_number,
                output_dir=str(output_dir),
                silent=True,
                verbose=self.verbose,
            )
            logger.create_dir()

        return SimulatedShopper(
            initial_message=experiment.prompt_template,
            engine_params=engine_params,
            environment=environment,
            logger=logger,
        )

    def create_experiment_request(
        self, experiment: ExperimentData, engine_params: EngineParams
    ) -> RawMessageExchange:
        agent = self._bootstrap(experiment, engine_params, persistence=False)

        return agent.get_batch_request()

    def process_experiment_result(
        self,
        experiment: ExperimentData,
        engine_params: EngineParams,
        result: ExperimentResult,
    ) -> bool:
        """Process an experiment result from a batch"""
        agent = self._bootstrap(experiment, engine_params, persistence=True)
        logger = agent.logger

        if not logger:
            raise ValueError("Logger not initialized")

        logger.record_cart_item(result.tool_call)

        tool_call_dict = {
            "name": "add_to_cart",
            "args": result.tool_call.model_dump(),
            "id": "batch_result",
        }
        msg = AIMessage(content=result.response_content, tool_calls=[tool_call_dict])

        logger.record_agent_interaction(msg)
        logger.finalize_journey_data()

        return True

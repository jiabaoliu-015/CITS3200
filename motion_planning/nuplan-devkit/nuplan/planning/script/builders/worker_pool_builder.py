import logging

from hydra.utils import instantiate
from omegaconf import DictConfig

from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool

logger = logging.getLogger(__name__)


def build_worker(cfg: DictConfig) -> WorkerPool:
    """
    Builds the worker.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Instance of WorkerPool.
    """
    logger.info('Building WorkerPool...')
    ray_worker_target = "nuplan.planning.utils.multithreading.worker_ray.RayDistributed"

    worker: WorkerPool = (
        instantiate(cfg.worker, output_dir=cfg.output_dir)
        if str(cfg.worker._target_) == ray_worker_target
        else instantiate(cfg.worker)
    )
    validate_type(worker, WorkerPool)

    logger.info('Building WorkerPool...DONE!')
    return worker

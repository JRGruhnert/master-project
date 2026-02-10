import multiprocessing as mp
from multiprocessing import Event, Queue
from concurrent.futures import ProcessPoolExecutor, as_completed

from scripts.train import TrainConfig
from src.factory import (
    select_agent,
    select_environment,
    select_evaluator,
    select_experiment,
)
from src.modules.buffer import Buffer
from src.modules.logger import Logger
from src.modules.storage import Storage


class ParallelTrainer:
    def __init__(self, config: TrainConfig, num_workers: int = 4):
        self.config = config
        self.num_workers = num_workers
        self.storage = Storage(config.storage)
        self.buffer = Buffer(config.buffer)  # Thread-safe or use Queue
        self.logger = Logger(config.logger)
        self.agent = select_agent(config.agent, self.storage, self.buffer)

    def collect_batch_parallel(self) -> bool:
        """Collect with multiple workers, each completes full episodes"""

        # Shared flag: "we have enough data, don't start new episodes"
        stop_flag = mp.Event()
        episode_queue = mp.Queue()

        # Target episodes per worker (rough split)
        episodes_needed = self.config.buffer.steps // self.num_workers + 1

        # Spawn workers
        workers = []
        for i in range(self.num_workers):
            p = mp.Process(
                target=self._worker_collect,
                args=(self.config, episode_queue, stop_flag, episodes_needed),
            )
            p.start()
            workers.append(p)

        # Main thread: consume episodes, check if batch ready
        collected = 0
        while not self._batch_ready():
            try:
                episode_data = episode_queue.get(timeout=1.0)
                self.buffer.add(episode_data)
                collected += 1

                # Signal workers to stop starting new episodes
                if self._batch_ready():
                    stop_flag.set()
            except:
                pass

        # Wait for workers to finish current episodes
        for p in workers:
            p.join()

        # Drain remaining episodes from queue
        while not episode_queue.empty():
            self.buffer.add(episode_queue.get_nowait())

        return True

    @staticmethod
    def _worker_collect(
        config: TrainConfig, queue: Queue, stop_flag: Event, max_episodes: int
    ):
        """Worker process: collect full episodes, check flag BETWEEN episodes"""

        # Each worker needs its own environment instance
        storage = Storage(config.storage)
        evaluator = select_evaluator(config.evaluator, storage)
        env = select_environment(config.environment, evaluator, storage)
        experiment = select_experiment(config.experiment, env, storage)
        agent = select_agent(config.agent, storage, None)  # No shared buffer

        episodes_done = 0

        while episodes_done < max_episodes:
            # CHECK BEFORE starting new episode (key part!)
            if stop_flag.is_set():
                break

            # Collect ONE full episode
            obs, goal = experiment.sample_task()
            episode_data = []
            episode_ended = False

            while not episode_ended:
                skill = agent.act(obs, goal)
                if skill:
                    obs, reward, done, episode_ended = experiment.step(skill)
                    episode_data.append((obs, reward, done, skill))

            # Put completed episode in queue
            queue.put(episode_data)
            episodes_done += 1

        experiment.close()

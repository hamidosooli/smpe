from collections import defaultdict
from hashlib import sha256
import json
import logging

import numpy as np


class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_wandb = False
        self.use_sacred = False
        self.use_hdf = False

        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value

        configure(directory_name)
        self.tb_logger = log_value
        self.use_tb = True

    def setup_wandb(self, config, team_name=None, project_name=None, mode="online"):
        import wandb

        assert mode in [
            "offline",
            "online",
        ], f"Invalid value for `wandb_mode`: {mode}. Supported values are 'online' and 'offline'."

        alg_name = config.get("name", "run")
        env_name = config.get("env", "env")
        env_args = config.get("env_args", {})
        if "map_name" in env_args:
            env_name += "_" + str(env_args["map_name"])
        elif "key" in env_args:
            env_name += "_" + str(env_args["key"])

        non_hash_keys = ["seed"]
        filtered_config = {k: v for k, v in config.items() if k not in non_hash_keys}
        self.config_hash = sha256(
            json.dumps(filtered_config, sort_keys=True, default=str).encode("utf8")
        ).hexdigest()[-10:]
        group_name = "_".join([alg_name, env_name, self.config_hash])

        init_kwargs = {
            "config": config,
            "group": group_name,
            "mode": mode,
        }
        if team_name:
            init_kwargs["entity"] = team_name
        if project_name:
            init_kwargs["project"] = project_name

        self.wandb = wandb.init(**init_kwargs)
        self.use_wandb = True

        self.console_logger.info("*******************")
        self.console_logger.info("WANDB RUN ID:")
        self.console_logger.info(f"{self.wandb.id}")
        self.console_logger.info("*******************")

        # Accumulate metrics at a timestep and flush once per step.
        self.wandb_current_t = -1
        self.wandb_current_data = {}

    def setup_sacred(self, sacred_run_dict):
        self._run_obj = sacred_run_dict
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_stat(self, key, value, t, to_sacred=True):
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_wandb:
            if self.wandb_current_t != t and self.wandb_current_data:
                self.wandb.log(self.wandb_current_data, step=self.wandb_current_t)
                self.wandb_current_data = {}
            self.wandb_current_t = t
            self.wandb_current_data[key] = value

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

            self._run_obj.log_scalar(key, value, t)

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(
            *self.stats["episode"][-1]
        )
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            try:
                item = "{:.4f}".format(np.mean([x[1] for x in self.stats[k][-window:]]))
            except:
                item = "{:.4f}".format(
                    np.mean([x[1].item() for x in self.stats[k][-window:]])
                )
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)

    def finish(self):
        if self.use_wandb:
            if self.wandb_current_data:
                self.wandb.log(self.wandb_current_data, step=self.wandb_current_t)
            self.wandb.finish()


# set up a custom logger
def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(levelname)s %(asctime)s] %(name)s %(message)s", "%H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel("DEBUG")

    return logger

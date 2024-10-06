import os

import torch

from config import get_args_dict


class CheckPoint:
    """
    Use this checkpoint class to load and save model, optimizer and gradient estimator.
    Goal is to be able to recreate exactly same result from checkpoint info
    Information includes:
        - model
                        - model_name, which is model.model_name, TODO: add model_name to all models
                        - model parameters dict, model.state_dict
        - optimizer
                        - name
                        - optimizer state dict, optimizer.state_dict
        - gradient estimator
                        - name, 'rge' or 'cge'
                        - gradient estimator state dict. rge.state_dict (NOTE: self-implemented)
        - rng_states
            - cpu state: tensor
            - cuda_all states: list[tensor]
                - training_history
                                - list of checkpoint step
                - training_history_since_last_checkpoint
                                - 1 checkpoint step: args & #epoches
                - lr_scheduler
                                - Last to implement, this feels not needed if we can resume training from checkpoint
    """

    def __init__(self, args, model, optimizer, gradient_estimator) -> None:
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.gradient_estimator = gradient_estimator

        self.history = []
        self.best_loss = None
        self.best_acc = None
        self.new_checkpoint_file_path = None

        checkpoint_file_path: str | None = args.checkpoint
        self.last_checkpoint_file = checkpoint_file_path
        if checkpoint_file_path is not None:
            try:
                checkpoint_data = torch.load(checkpoint_file_path)
            except FileNotFoundError:
                raise Exception("Fail to load checkpoint")
        else:
            checkpoint_data = None

        if checkpoint_data is None:
            return

        self.history = checkpoint_data["history"]
        # model
        self.load_model(checkpoint_data.get("model"))

        # optimizer, TODO: discuss if we need to load momentum buffer
        self.load_optimizer(checkpoint_data.get("optimizer"))

        # random seed, TODO: improve later, this does not work at this moment, investigate when have time
        # self.load_rng_states(checkpoint_data.get("rng_states"))

        # gradient estimator
        # ge_dict = checkpoint_data.get("gradient_estimator")
        # if self.gradient_estimator.__class__.__name__ == ge_dict["name"]:
        #     self.gradient_estimator.load_state_dict(ge_dict["state_dict"])
        #     # TODO: update gradient_estimator's parameter using args
        # else:
        #     raise Exception("Optimizer does not match checkpoint!")

        # TODO: LR scheduler

    def get_trained_epochs(self):
        ret = 0
        for hist in self.history:
            ret += hist["n_epoch"]
        return ret

    def load_model(self, model_dict):
        if self.model.model_name == model_dict["model_name"]:
            self.model.load_state_dict(model_dict["state_dict"])
        else:
            raise Exception("Model does not match checkpoint!")

    def load_rng_states(self, rng_states):
        print(f"Loading RNG from checkpoint, thus igoring random seed {self.args.seed}")
        torch.random.set_rng_state(rng_states["cpu"])
        torch.cuda.set_rng_state_all(rng_states["cuda_all"])

    def load_optimizer(self, optimizer_dict):
        if self.optimizer.__class__.__name__ == optimizer_dict["name"]:
            optimizer_state_dict = optimizer_dict["state_dict"]
            # update lr
            optimizer_state_dict["param_groups"][0].update(
                {
                    "lr": self.args.lr,
                    "initial_lr": self.args.lr,
                }
            )
            self.optimizer.load_state_dict(optimizer_state_dict)
        else:
            raise Exception("Optimizer does not match checkpoint!")

    def _generate_save_data(self, file_name, epoch_idx):
        checkpoint_step = {"n_epoch": epoch_idx + 1, "args": get_args_dict(self.args)}
        return {
            "model": {
                "model_name": self.model.model_name,
                "state_dict": self.model.state_dict(),
            },
            "optimizer": {
                "name": self.optimizer.__class__.__name__,
                "state_dict": self.optimizer.state_dict(),
            },
            # "gradient_estimator": {
            #     "name": self.gradient_estimator.__class__.__name__,
            #     "state_dict": self.gradient_estimator.state_dict(),
            # },
            "rng_states": {
                "cpu": torch.random.get_rng_state(),
                "cuda_all": torch.cuda.get_rng_state_all(),
            },
            "last_checkpoint": self.last_checkpoint_file,
            "history": self.history + [checkpoint_step],
            "checkpoint_step_since_last_checkpoint": checkpoint_step,
        }

    def should_update(self, eval_loss, eval_acc, epoch_idx):
        update_plan = self.args.checkpoint_update_plan
        if update_plan == "never":
            return False

        if update_plan in ["every5", "every10"]:
            update_freq = {"every5": 5, "every10": 10}.get(update_plan)
            if ((epoch_idx + 1) % update_freq) == 0:
                return True

        if update_plan == "best_loss":
            if self.best_loss is None or eval_loss < self.best_loss:
                self.best_loss = eval_loss
                return True

        if update_plan == "best_acc":
            if self.best_acc is None or eval_acc > self.best_acc:
                self.best_acc = eval_acc
                return True

        return False

    def save(self, file_name, epoch_idx, subfolder=None):
        to_save = self._generate_save_data(file_name, epoch_idx)

        if subfolder:
            folder_path = "./checkpoints/" + subfolder + "/"
        else:
            folder_path = "./checkpoints/"

        os.makedirs(folder_path, exist_ok=True)

        new_file_path = folder_path + file_name + ".pth"
        if self.args.create_many_checkpoint:
            file_path = new_file_path

        else:
            if self.new_checkpoint_file_path is None:
                self.new_checkpoint_file_path = new_file_path

            file_path = self.new_checkpoint_file_path

        print(f"Saving checkpoint {file_path}")
        torch.save(to_save, file_path)

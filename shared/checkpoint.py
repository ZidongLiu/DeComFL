from typing import Optional
import torch
import os

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
        - random seed dict
                        - random seed
                        - number of perturbation/random_pruning_mask generated using this seed(tricky)
                - training_history
                                - list of checkpoint step
                - training_history_since_last_checkpoint
                                - 1 checkpoint step: args & #epoches
                - lr_scheduler
                                - Last to implement, this feels not needed if we can resume training from checkpoint
    """

    def __init__(self, args, model, optimizer, gradient_estimator):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.gradient_estimator = gradient_estimator

        self.history = []
        self.best_loss = None
        self.best_acc = None

        checkpoint_file_path: Optional[str] = args.checkpoint
        self.last_checkpoint_file = checkpoint_file_path
        if checkpoint_file_path is not None:
            try:
                checkpoint_data = torch.load(checkpoint_file_path)
            except:
                raise Exception("Fail to load ")
        else:
            checkpoint_data = None

        if checkpoint_data is None:
            return

        self.history = checkpoint_data["history"]
        # model
        model_dict = checkpoint_data.get("model")
        if self.model.model_name == model_dict["model_name"]:
            self.model.load_state_dict(model_dict["state_dict"])
        else:
            raise Exception("Model does not match checkpoint!")

        # optimizer
        optimizer_dict = checkpoint_data.get("optimizer")
        if self.optimizer.__class__.__name__ == optimizer_dict["name"]:
            self.optimizer.load_state_dict(optimizer_dict["state_dict"])
            # TODO: update optimizer's parameter using args
        else:
            raise Exception("Optimizer does not match checkpoint!")

        # gradient estimator
        # ge_dict = checkpoint_data.get("gradient_estimator")
        # if self.gradient_estimator.__class__.__name__ == ge_dict["name"]:
        #     self.gradient_estimator.load_state_dict(ge_dict["state_dict"])
        #     # TODO: update gradient_estimator's parameter using args
        # else:
        #     raise Exception("Optimizer does not match checkpoint!")

        # TODO: random seed

    def _generate_save_data(self, file_name, n_epoch):
        checkpoint_step = {"n_epoch": n_epoch, "args": get_args_dict(self.args)}
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
            # "random_seed": {
            #     "seed": self.args.seed,
            #     "count": n_epoch,
            # },
            "last_checkpoint": self.last_checkpoint_file,
            "history": self.history + [checkpoint_step],
            "checkpoint_step_since_last_checkpoint": checkpoint_step,
        }

    def should_update(self, eval_loss, eval_acc, n_epoch):
        update_plan = self.args.checkpoint_update_plan
        if update_plan == "never":
            return

        if update_plan in ["every5", "every10"]:
            update_freq = {"every5": 5, "every10": 10}.get(update_plan)
            if (n_epoch % update_freq) == (update_freq - 1):
                return True

        if update_plan == "best_loss":
            if self.best_loss is None or eval_loss > self.best_loss:
                self.best_loss = eval_loss
                return True

        if update_plan == "best_acc":
            if self.best_acc is None or eval_acc > self.best_acc:
                self.best_acc = eval_acc
                return True

        return False

    def save(self, file_name, n_epoch):
        to_save = self._generate_save_data(file_name, n_epoch)

        os.makedirs("checkpoints/", exist_ok=True)

        new_file_path = "./checkpoints/" + file_name + ".pth"
        if self.args.checkpoint_overwrite:
            if self.last_checkpoint_file is None:
                self.last_checkpoint_file = new_file_path

            file_path = self.last_checkpoint_file
        else:
            file_path = new_file_path

        torch.save(to_save, file_path)

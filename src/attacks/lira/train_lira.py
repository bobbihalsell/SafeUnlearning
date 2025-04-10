from utils import get_retain_forget_val_test_indices
from pipeline.step_5_unlearn import UnlearnerApp
from munl.datasets import get_loaders_from_dataset_and_unlearner_from_cfg_with_indices
from pathlib import Path
import numpy as np


def main(args):
    unlearner_name = args.unlearner_name
    model_name = args.model
    dataset_name = args.dataset
    root = args.root
    seed = args.seed
    num_splits = args.num_splits
    num_forgets = args.num_forgets

    name_save_path = "artifacts/lira/unlearn"
    if not Path(name_save_path).exists():
        Path(name_save_path).mkdir(parents=True, exist_ok=True)
    # from args
    # dataset_cfg, unlearner_cfg, model_cfg

    for split_ndx in range(num_splits):
        for forget_ndx in range(num_forgets):
            retain, forget, val, test = get_retain_forget_val_test_indices(
                lira_path=root / "artifacts" / "lira" / "splits",
                split_ndx=split_ndx,
                forget_ndx=forget_ndx,
            )

            train_loader, retain_loader, forget_loader, val_loader, test_loader = (
                get_loaders_from_dataset_and_unlearner_from_cfg_with_indices(
                    root=root,
                    indices=[np.concatenate([retain, forget]), retain, forget, val, test],
                    dataset_cfg=dataset_cfg,
                    unlearner_cfg=unlearner.cfg,
                )
            )

            model = app.get_model(root / "artifacts" / dataset_name)
            original_model, unlearned_model = app.run_from_model_and_loaders(
                root=root,
                model=model,
                loaders=(train_loader, retain_loader, forget_loader, val_loader, test_loader),
                save=True,
                save_path=name_save_path,
                save_name=f"{split_ndx}_{forget_ndx}",
                unlearner_name=unlearner_name,
            )

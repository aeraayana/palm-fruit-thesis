from autogluon.multimodal.utils.misc import shopee_dataset
from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.presets import get_automm_presets
from IPython.display import Image, display
from ray import tune
from datetime import datetime
import json
import yaml
import uuid

def main():
    download_dir = './ag_automm_tutorial_imgcls'
    train_data_path, test_data_path = shopee_dataset(download_dir)

    model_path = f"./tmp/{uuid.uuid4().hex}-automm_shopee"

    predictor_hpo = MultiModalPredictor(label="label", eval_metric="f1_macro", path=model_path)
    predictor_hpo.set_verbosity(3)

    hyperparameters = {
        "optimization.learning_rate": tune.uniform(0.00005, 0.001),
        "model.timm_image.checkpoint_name": tune.choice(["ghostnet_100", "mobilenetv3_large_100"])
    }

    hyperparameter_tune_kwargs = {
        "searcher": "bayes",
        "scheduler": "ASHA",
        "num_trials": 5,
        "num_to_keep": 5,
    }

    start_time_hpo = datetime.now()

    predictor_hpo.fit(
        train_data=train_data_path,
        hyperparameters=hyperparameters,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
    )

if __name__ == '__main__':
    main()

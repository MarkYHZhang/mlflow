import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np
import mlflow
import matplotlib.pyplot as plt
from mlflow.models.evaluation.artifacts import CsvEvaluationArtifact

# loading the California housing dataset
cali_housing = fetch_california_housing(as_frame=True)

# split the dataset into train and test partitions
X_train, X_test, y_train, y_test = train_test_split(
    cali_housing.data, cali_housing.target, test_size=0.2, random_state=123
)

# train the model
lin_reg = LinearRegression().fit(X_train, y_train)

# creating the evaluation dataframe
eval_data = X_test.copy()
eval_data["target"] = y_test


def metrics_only_fn(eval_df, builtin_metrics):
    """
    This example demonstrates an example custom metric function that does not
    produce any artifacts. Also notice that for computing its metrics, it can either
    directly use the eval_df or build upon existing metrics supplied by builtin_metrics
    """
    return {
        "squared_diff_plus_one": np.sum(np.abs(eval_df["prediction"] - eval_df["target"] + 1) ** 2),
        "sum_on_label_divided_by_two": builtin_metrics["sum_on_label"] / 2,
    }


def file_artifacts_fn(eval_df, builtin_metrics):
    """
    This example shows how you can return file paths as representation
    of the produced artifacts. For a full list of supported file extensions
    refer to https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate
    :param eval_df:
    :param builtin_metrics:
    :return:
    """
    example_np_arr = np.array([1, 2, 3])
    np.save("example.npy", example_np_arr, allow_pickle=False)

    example_df = pd.DataFrame({"test": [2.2, 3.1], "test2": [3, 2]})
    example_df.to_csv("example.csv", index=False)
    example_df.to_parquet("example.parquet")

    example_json = {"hello": "there", "test_list": [0.1, 0.3, 4]}
    example_json.update(builtin_metrics)
    with open("example.json", "w") as f:
        json.dump(example_json, f)

    plt.scatter(eval_df["prediction"], eval_df["target"])
    plt.xlabel("Targets")
    plt.ylabel("Predictions")
    plt.title("Targets vs. Predictions")
    plt.savefig("example.png")
    plt.savefig("example.jpeg")

    return {}, {
        "example_npy": "example.npy",
        "example_csv": "example.csv",
        "example_parquet": "example.parquet",
        "example_json": "example.json",
        "example_png": "example.png",
        "example_jpg": "example.jpeg",
    }


def object_artifacts_fn(eval_df, builtin_metrics):
    """
    This example shows how you can return python objects as artifacts
    without the need to save them to file system.
    """
    example_np_arr = np.array([1, 2, 3])
    example_df = pd.DataFrame({"test": [2.2, 3.1], "test2": [3, 2]})
    example_dict = {"hello": "there", "test_list": [0.1, 0.3, 4]}
    example_dict.update(builtin_metrics)
    example_image = plt.figure()
    plt.scatter(eval_df["prediction"], eval_df["target"])
    plt.xlabel("Targets")
    plt.ylabel("Predictions")
    plt.title("Targets vs. Predictions")

    # In rare cases, you might already have constructed a EvaluationArtifact-like object
    # in which case, you can directly return that as well.
    example_df.to_csv("example2.csv", index=False)
    mlflow.log_artifact("example2.csv")
    csv_eval_artifact = CsvEvaluationArtifact(uri=mlflow.get_artifact_uri("example2.csv"))

    return {}, {
        "example_np_arr": example_np_arr,
        "example_df": example_df,
        "example_dict": example_dict,
        "example_image": example_image,
        "example_csv_artifact": csv_eval_artifact,
    }


def mixed_example_fn(eval_df, builtin_metrics):
    """
    This example mixes together some of the different ways to return metrics and artifacts
    """
    metrics = {
        "squared_diff_divided_two": np.sum(
            np.abs(eval_df["prediction"] - eval_df["target"]) ** 2 / 2
        ),
        "sum_on_label_multiplied_by_three": builtin_metrics["sum_on_label"] * 3,
    }
    example_dict = {"hello": "there", "test_list": [0.1, 0.3, 4]}
    example_dict.update(builtin_metrics)
    plt.scatter(eval_df["prediction"], eval_df["target"])
    plt.xlabel("Targets")
    plt.ylabel("Predictions")
    plt.title("Targets vs. Predictions")
    plt.savefig("example2.png")
    artifacts = {
        "example_dict_2": example_dict,
        "example_image_2": "example2.png",
    }
    return metrics, artifacts


with mlflow.start_run() as run:
    mlflow.sklearn.log_model(lin_reg, "model")
    model_uri = mlflow.get_artifact_uri("model")
    result = mlflow.evaluate(
        model=model_uri,
        data=eval_data,
        targets="target",
        model_type="regressor",
        dataset_name="cali_housing",
        evaluators=["default"],
        custom_metrics=[
            metrics_only_fn,
            file_artifacts_fn,
            object_artifacts_fn,
            mixed_example_fn,
        ],
    )

print(f"metrics:\n{result.metrics}")
print(f"artifacts:\n{result.artifacts}")

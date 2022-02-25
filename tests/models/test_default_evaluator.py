import numpy as np
import json
import pandas as pd
import pytest
from contextlib import nullcontext as does_not_raise

from mlflow.exceptions import MlflowException
from mlflow.models.evaluation import evaluate
from mlflow.models.evaluation.artifacts import (
    CsvEvaluationArtifact,
    ImageEvaluationArtifact,
    JsonEvaluationArtifact,
    NumpyEvaluationArtifact,
    ParquetEvaluationArtifact,
)
from mlflow.models.evaluation.default_evaluator import (
    _get_classifier_global_metrics,
    _infer_model_type_by_labels,
    _extract_raw_model_and_predict_fn,
    _get_regressor_metrics,
    _get_binary_sum_up_label_pred_prob,
    _get_classifier_per_class_metrics,
    _gen_classifier_curve,
    _evaluate_custom_metric,
    _load_custom_metric_artifact,
)
import mlflow
from sklearn.linear_model import LogisticRegression

# pylint: disable=unused-import
from mlflow.utils.file_utils import TempDir
from tests.models.test_evaluation import (
    get_run_data,
    linear_regressor_model_uri,
    diabetes_dataset,
    multiclass_logistic_regressor_model_uri,
    iris_dataset,
    binary_logistic_regressor_model_uri,
    breast_cancer_dataset,
    spark_linear_regressor_model_uri,
    diabetes_spark_dataset,
    svm_model_uri,
)
from mlflow.models.utils import plot_lines


def assert_dict_equal(d1, d2, rtol):
    for k in d1:
        assert k in d2
        assert np.isclose(d1[k], d2[k], rtol=rtol)


def test_regressor_evaluation(linear_regressor_model_uri, diabetes_dataset):
    with mlflow.start_run() as run:
        result = evaluate(
            linear_regressor_model_uri,
            diabetes_dataset._constructor_args["data"],
            model_type="regressor",
            targets=diabetes_dataset._constructor_args["targets"],
            dataset_name=diabetes_dataset.name,
            evaluators="default",
        )

    _, metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(linear_regressor_model_uri)

    y = diabetes_dataset.labels_data
    y_pred = model.predict(diabetes_dataset.features_data)

    expected_metrics = _get_regressor_metrics(y, y_pred)
    for metric_key in expected_metrics:
        assert np.isclose(
            expected_metrics[metric_key],
            metrics[metric_key + "_on_data_diabetes_dataset"],
            rtol=1e-3,
        )
        assert np.isclose(expected_metrics[metric_key], result.metrics[metric_key], rtol=1e-3)

    assert json.loads(tags["mlflow.datasets"]) == [
        {**diabetes_dataset._metadata, "model": model.metadata.model_uuid}
    ]

    assert set(artifacts) == {
        "shap_beeswarm_plot_on_data_diabetes_dataset.png",
        "shap_feature_importance_plot_on_data_diabetes_dataset.png",
        "shap_summary_plot_on_data_diabetes_dataset.png",
    }
    assert result.artifacts.keys() == {
        "shap_beeswarm_plot",
        "shap_feature_importance_plot",
        "shap_summary_plot",
    }


def test_multi_classifier_evaluation(multiclass_logistic_regressor_model_uri, iris_dataset):
    with mlflow.start_run() as run:
        result = evaluate(
            multiclass_logistic_regressor_model_uri,
            iris_dataset._constructor_args["data"],
            model_type="classifier",
            targets=iris_dataset._constructor_args["targets"],
            dataset_name=iris_dataset.name,
            evaluators="default",
        )

    _, metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(multiclass_logistic_regressor_model_uri)

    _, _, predict_fn, predict_proba_fn = _extract_raw_model_and_predict_fn(model)
    y = iris_dataset.labels_data
    y_pred = predict_fn(iris_dataset.features_data)
    y_probs = predict_proba_fn(iris_dataset.features_data)

    expected_metrics = _get_classifier_global_metrics(False, y, y_pred, y_probs, labels=None)

    for metric_key in expected_metrics:
        assert np.isclose(
            expected_metrics[metric_key], metrics[metric_key + "_on_data_iris_dataset"], rtol=1e-3
        )
        assert np.isclose(expected_metrics[metric_key], result.metrics[metric_key], rtol=1e-3)

    assert json.loads(tags["mlflow.datasets"]) == [
        {**iris_dataset._metadata, "model": model.metadata.model_uuid}
    ]

    assert set(artifacts) == {
        "shap_beeswarm_plot_on_data_iris_dataset.png",
        "per_class_metrics_on_data_iris_dataset.csv",
        "roc_curve_plot_on_data_iris_dataset.png",
        "precision_recall_curve_plot_on_data_iris_dataset.png",
        "shap_feature_importance_plot_on_data_iris_dataset.png",
        "explainer_on_data_iris_dataset",
        "confusion_matrix_on_data_iris_dataset.png",
        "shap_summary_plot_on_data_iris_dataset.png",
    }
    assert result.artifacts.keys() == {
        "per_class_metrics",
        "roc_curve_plot",
        "precision_recall_curve_plot",
        "confusion_matrix",
        "shap_beeswarm_plot",
        "shap_summary_plot",
        "shap_feature_importance_plot",
    }


def test_bin_classifier_evaluation(binary_logistic_regressor_model_uri, breast_cancer_dataset):
    with mlflow.start_run() as run:
        result = evaluate(
            binary_logistic_regressor_model_uri,
            breast_cancer_dataset._constructor_args["data"],
            model_type="classifier",
            targets=breast_cancer_dataset._constructor_args["targets"],
            dataset_name=breast_cancer_dataset.name,
            evaluators="default",
        )

    _, metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(binary_logistic_regressor_model_uri)

    _, _, predict_fn, predict_proba_fn = _extract_raw_model_and_predict_fn(model)
    y = breast_cancer_dataset.labels_data
    y_pred = predict_fn(breast_cancer_dataset.features_data)
    y_probs = predict_proba_fn(breast_cancer_dataset.features_data)

    expected_metrics = _get_classifier_global_metrics(True, y, y_pred, y_probs, labels=None)

    for metric_key in expected_metrics:
        assert np.isclose(
            expected_metrics[metric_key],
            metrics[metric_key + "_on_data_breast_cancer_dataset"],
            rtol=1e-3,
        )
        assert np.isclose(expected_metrics[metric_key], result.metrics[metric_key], rtol=1e-3)

    assert json.loads(tags["mlflow.datasets"]) == [
        {**breast_cancer_dataset._metadata, "model": model.metadata.model_uuid}
    ]

    assert set(artifacts) == {
        "shap_feature_importance_plot_on_data_breast_cancer_dataset.png",
        "lift_curve_plot_on_data_breast_cancer_dataset.png",
        "shap_beeswarm_plot_on_data_breast_cancer_dataset.png",
        "precision_recall_curve_plot_on_data_breast_cancer_dataset.png",
        "confusion_matrix_on_data_breast_cancer_dataset.png",
        "shap_summary_plot_on_data_breast_cancer_dataset.png",
        "roc_curve_plot_on_data_breast_cancer_dataset.png",
    }
    assert result.artifacts.keys() == {
        "roc_curve_plot",
        "precision_recall_curve_plot",
        "lift_curve_plot",
        "confusion_matrix",
        "shap_beeswarm_plot",
        "shap_summary_plot",
        "shap_feature_importance_plot",
    }


def test_spark_regressor_model_evaluation(spark_linear_regressor_model_uri, diabetes_spark_dataset):
    with mlflow.start_run() as run:
        result = evaluate(
            spark_linear_regressor_model_uri,
            diabetes_spark_dataset._constructor_args["data"],
            model_type="regressor",
            targets=diabetes_spark_dataset._constructor_args["targets"],
            dataset_name=diabetes_spark_dataset.name,
            evaluators="default",
            evaluator_config={"log_model_explainability": True},
        )

    _, metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(spark_linear_regressor_model_uri)

    X = diabetes_spark_dataset.features_data
    y = diabetes_spark_dataset.labels_data
    y_pred = model.predict(X)

    expected_metrics = _get_regressor_metrics(y, y_pred)

    for metric_key in expected_metrics:
        assert np.isclose(
            expected_metrics[metric_key],
            metrics[metric_key + "_on_data_diabetes_spark_dataset"],
            rtol=1e-3,
        )
        assert np.isclose(expected_metrics[metric_key], result.metrics[metric_key], rtol=1e-3)

    model = mlflow.pyfunc.load_model(spark_linear_regressor_model_uri)

    assert json.loads(tags["mlflow.datasets"]) == [
        {**diabetes_spark_dataset._metadata, "model": model.metadata.model_uuid}
    ]

    assert set(artifacts) == set()
    assert result.artifacts == {}


def test_svm_classifier_evaluation(svm_model_uri, breast_cancer_dataset):
    with mlflow.start_run() as run:
        result = evaluate(
            svm_model_uri,
            breast_cancer_dataset._constructor_args["data"],
            model_type="classifier",
            targets=breast_cancer_dataset._constructor_args["targets"],
            dataset_name=breast_cancer_dataset.name,
            evaluators="default",
        )

    _, metrics, tags, artifacts = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(svm_model_uri)

    _, _, predict_fn, _ = _extract_raw_model_and_predict_fn(model)
    y = breast_cancer_dataset.labels_data
    y_pred = predict_fn(breast_cancer_dataset.features_data)

    expected_metrics = _get_classifier_global_metrics(True, y, y_pred, None, labels=None)

    for metric_key in expected_metrics:
        assert np.isclose(
            expected_metrics[metric_key],
            metrics[metric_key + "_on_data_breast_cancer_dataset"],
            rtol=1e-3,
        )
        assert np.isclose(expected_metrics[metric_key], result.metrics[metric_key], rtol=1e-3)

    assert json.loads(tags["mlflow.datasets"]) == [
        {**breast_cancer_dataset._metadata, "model": model.metadata.model_uuid}
    ]

    assert set(artifacts) == {
        "confusion_matrix_on_data_breast_cancer_dataset.png",
        "shap_feature_importance_plot_on_data_breast_cancer_dataset.png",
        "shap_beeswarm_plot_on_data_breast_cancer_dataset.png",
        "shap_summary_plot_on_data_breast_cancer_dataset.png",
    }
    assert result.artifacts.keys() == {
        "confusion_matrix",
        "shap_beeswarm_plot",
        "shap_summary_plot",
        "shap_feature_importance_plot",
    }


def test_infer_model_type_by_labels():
    assert _infer_model_type_by_labels(["a", "b"]) == "classifier"
    assert _infer_model_type_by_labels([1, 2.5]) == "regressor"
    assert _infer_model_type_by_labels(list(range(2000))) == "regressor"
    assert _infer_model_type_by_labels([1, 2, 3]) == "classifier"


def test_extract_raw_model_and_predict_fn(binary_logistic_regressor_model_uri):
    model = mlflow.pyfunc.load_model(binary_logistic_regressor_model_uri)
    (
        model_loader_module,
        raw_model,
        predict_fn,
        predict_proba_fn,
    ) = _extract_raw_model_and_predict_fn(model)
    assert model_loader_module == "mlflow.sklearn"
    assert isinstance(raw_model, LogisticRegression)
    assert predict_fn == raw_model.predict
    assert predict_proba_fn == raw_model.predict_proba


def test_get_regressor_metrics():
    y = [1.1, 2.1, -3.5]
    y_pred = [1.5, 2.0, -3.0]

    metrics = _get_regressor_metrics(y, y_pred)
    expected_metrics = {
        "example_count": 3,
        "mean_absolute_error": 0.3333333333333333,
        "mean_squared_error": 0.13999999999999999,
        "root_mean_squared_error": 0.3741657386773941,
        "sum_on_label": -0.2999999999999998,
        "mean_on_label": -0.09999999999999994,
        "r2_score": 0.976457399103139,
        "max_error": 0.5,
        "mean_absolute_percentage_error": 0.18470418470418468,
    }
    assert_dict_equal(metrics, expected_metrics, rtol=1e-3)


def test_get_binary_sum_up_label_pred_prob():
    y = [0, 1, 2]
    y_pred = [0, 2, 1]
    y_probs = [[0.7, 0.1, 0.2], [0.2, 0.3, 0.5], [0.25, 0.4, 0.35]]

    results = []
    for idx, label in enumerate([0, 1, 2]):
        y_bin, y_pred_bin, y_prob_bin = _get_binary_sum_up_label_pred_prob(
            idx, label, y, y_pred, y_probs
        )
        results.append((list(y_bin), list(y_pred_bin), list(y_prob_bin)))

    print(results)
    assert results == [
        ([1, 0, 0], [1, 0, 0], [0.7, 0.2, 0.25]),
        ([0, 1, 0], [0, 0, 1], [0.1, 0.3, 0.4]),
        ([0, 0, 1], [0, 1, 0], [0.2, 0.5, 0.35]),
    ]


def test_get_classifier_per_class_metrics():
    y = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
    y_pred = [0, 1, 1, 0, 1, 1, 0, 1, 1, 0]

    expected_metrics = {
        "true_negatives": 3,
        "false_positives": 2,
        "false_negatives": 1,
        "true_positives": 4,
        "recall": 0.8,
        "precision": 0.6666666666666666,
        "f1_score": 0.7272727272727272,
    }
    metrics = _get_classifier_per_class_metrics(y, y_pred)
    assert_dict_equal(metrics, expected_metrics, rtol=1e-3)


def test_multiclass_get_classifier_global_metrics():
    y = [0, 1, 2, 1, 2]
    y_pred = [0, 2, 1, 1, 0]
    y_probs = [
        [0.7, 0.1, 0.2],
        [0.2, 0.3, 0.5],
        [0.25, 0.4, 0.35],
        [0.3, 0.4, 0.3],
        [0.8, 0.1, 0.1],
    ]

    metrics = _get_classifier_global_metrics(
        is_binomial=False, y=y, y_pred=y_pred, y_probs=y_probs, labels=[0, 1, 2]
    )
    expected_metrics = {
        "accuracy": 0.4,
        "example_count": 5,
        "f1_score_micro": 0.4,
        "f1_score_macro": 0.38888888888888884,
        "log_loss": 1.1658691395263094,
    }
    assert_dict_equal(metrics, expected_metrics, 1e-3)


def test_binary_get_classifier_global_metrics():
    y = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
    y_pred = [0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
    y_prob = [0.1, 0.9, 0.8, 0.2, 0.7, 0.8, 0.3, 0.6, 0.65, 0.4]
    y_probs = [[1 - p, p] for p in y_prob]
    metrics = _get_classifier_global_metrics(
        is_binomial=True, y=y, y_pred=y_pred, y_probs=y_probs, labels=[0, 1]
    )
    expected_metrics = {"accuracy": 0.7, "example_count": 10, "log_loss": 0.6665822319387167}
    assert_dict_equal(metrics, expected_metrics, 1e-3)


def test_gen_binary_precision_recall_curve():
    y = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
    y_prob = [0.1, 0.9, 0.8, 0.2, 0.7, 0.8, 0.3, 0.6, 0.65, 0.4]

    results = _gen_classifier_curve(
        is_binomial=True, y=y, y_probs=y_prob, labels=[0, 1], curve_type="pr"
    )
    assert np.allclose(
        results.plot_fn_args["data_series"][0][1],
        np.array([1.0, 0.8, 0.8, 0.8, 0.6, 0.4, 0.4, 0.2, 0.0]),
        rtol=1e-3,
    )
    assert np.allclose(
        results.plot_fn_args["data_series"][0][2],
        np.array([0.55555556, 0.5, 0.57142857, 0.66666667, 0.6, 0.5, 0.66666667, 1.0, 1.0]),
        rtol=1e-3,
    )
    assert results.plot_fn_args["xlabel"] == "recall"
    assert results.plot_fn_args["ylabel"] == "precision"
    assert results.plot_fn_args["line_kwargs"] == {"drawstyle": "steps-post", "linewidth": 1}
    assert np.isclose(results.auc, 0.7088888888888889, rtol=1e-3)


def test_gen_binary_roc_curve():
    y = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
    y_prob = [0.1, 0.9, 0.8, 0.2, 0.7, 0.8, 0.3, 0.6, 0.65, 0.4]

    results = _gen_classifier_curve(
        is_binomial=True, y=y, y_probs=y_prob, labels=[0, 1], curve_type="roc"
    )
    assert np.allclose(
        results.plot_fn_args["data_series"][0][1],
        np.array([0.0, 0.0, 0.2, 0.4, 0.4, 0.8, 0.8, 1.0]),
        rtol=1e-3,
    )
    assert np.allclose(
        results.plot_fn_args["data_series"][0][2],
        np.array([0.0, 0.2, 0.4, 0.4, 0.8, 0.8, 1.0, 1.0]),
        rtol=1e-3,
    )
    assert results.plot_fn_args["xlabel"] == "False Positive Rate"
    assert results.plot_fn_args["ylabel"] == "True Positive Rate"
    assert results.plot_fn_args["line_kwargs"] == {"drawstyle": "steps-post", "linewidth": 1}
    assert np.isclose(results.auc, 0.66, rtol=1e-3)


def test_gen_multiclass_precision_recall_curve():
    y = [0, 1, 2, 1, 2]
    y_probs = [
        [0.7, 0.1, 0.2],
        [0.2, 0.3, 0.5],
        [0.25, 0.4, 0.35],
        [0.3, 0.4, 0.3],
        [0.8, 0.1, 0.1],
    ]

    results = _gen_classifier_curve(
        is_binomial=False, y=y, y_probs=y_probs, labels=[0, 1, 2], curve_type="pr"
    )
    expected_x_data_list = [[1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [1.0, 0.5, 0.5, 0.5, 0.0, 0.0]]
    expected_y_data_list = [
        [0.5, 0.0, 1.0],
        [0.66666667, 0.5, 1.0],
        [0.4, 0.25, 0.33333333, 0.5, 0.0, 1.0],
    ]
    line_labels = ["label=0,AP=0.500", "label=1,AP=0.722", "label=2,AP=0.414"]
    for index, (name, x_data, y_data) in enumerate(results.plot_fn_args["data_series"]):
        assert name == line_labels[index]
        assert np.allclose(x_data, expected_x_data_list[index], rtol=1e-3)
        assert np.allclose(y_data, expected_y_data_list[index], rtol=1e-3)

    assert results.plot_fn_args["xlabel"] == "recall"
    assert results.plot_fn_args["ylabel"] == "precision"
    assert results.plot_fn_args["line_kwargs"] == {"drawstyle": "steps-post", "linewidth": 1}

    expected_auc = [0.25, 0.6666666666666666, 0.2875]
    assert np.allclose(results.auc, expected_auc, rtol=1e-3)


def test_gen_multiclass_roc_curve():
    y = [0, 1, 2, 1, 2]
    y_probs = [
        [0.7, 0.1, 0.2],
        [0.2, 0.3, 0.5],
        [0.25, 0.4, 0.35],
        [0.3, 0.4, 0.3],
        [0.8, 0.1, 0.1],
    ]

    results = _gen_classifier_curve(
        is_binomial=False, y=y, y_probs=y_probs, labels=[0, 1, 2], curve_type="roc"
    )
    print(results)

    expected_x_data_list = [
        [0.0, 0.25, 0.25, 1.0],
        [0.0, 0.33333333, 0.33333333, 1.0],
        [0.0, 0.33333333, 0.33333333, 1.0, 1.0],
    ]
    expected_y_data_list = [[0.0, 0.0, 1.0, 1.0], [0.0, 0.5, 1.0, 1.0], [0.0, 0.0, 0.5, 0.5, 1.0]]
    line_labels = ["label=0,AUC=0.750", "label=1,AUC=0.750", "label=2,AUC=0.333"]
    for index, (name, x_data, y_data) in enumerate(results.plot_fn_args["data_series"]):
        assert name == line_labels[index]
        assert np.allclose(x_data, expected_x_data_list[index], rtol=1e-3)
        assert np.allclose(y_data, expected_y_data_list[index], rtol=1e-3)

    assert results.plot_fn_args["xlabel"] == "False Positive Rate"
    assert results.plot_fn_args["ylabel"] == "True Positive Rate"
    assert results.plot_fn_args["line_kwargs"] == {"drawstyle": "steps-post", "linewidth": 1}

    expected_auc = [0.75, 0.75, 0.3333]
    assert np.allclose(results.auc, expected_auc, rtol=1e-3)


def test_evaluate_custom_metric_incorrect_return_formats():
    eval_df = pd.DataFrame({"prediction": [1.2, 1.9, 3.2], "target": [1, 2, 3]})
    metrics = _get_regressor_metrics(eval_df["target"], eval_df["prediction"])

    def dummy_fn(*_):
        pass

    with pytest.raises(
        MlflowException,
        match=f"'{dummy_fn.__name__}' (.*) returned None",
    ):
        _evaluate_custom_metric(0, dummy_fn, eval_df, metrics)

    def incorrect_return_type_1(*_):
        return 3

    def incorrect_return_type_2(*_):
        return "stuff", 3

    for test_fn in (
        incorrect_return_type_1,
        incorrect_return_type_2,
    ):
        with pytest.raises(
            MlflowException,
            match=f"'{test_fn.__name__}' (.*) did not return in an expected format",
        ):
            _evaluate_custom_metric(0, test_fn, eval_df, metrics)

    def non_str_metric_name(*_):
        return {123: 123, "a": 32.1, "b": 3}

    def non_numerical_metric_value(*_):
        return {"stuff": 12, "non_numerical_metric": "123"}

    for test_fn in (
        non_str_metric_name,
        non_numerical_metric_value,
    ):
        with pytest.raises(
            MlflowException,
            match=f"'{test_fn.__name__}' (.*) did not return metrics as a dictionary of "
            "string metric names with numerical values",
        ):
            _evaluate_custom_metric(0, test_fn, eval_df, metrics)

    def non_str_artifact_name(*_):
        return {"a": 32.1, "b": 3}, {1: [1, 2, 3]}

    with pytest.raises(
        MlflowException,
        match=f"'{non_str_artifact_name.__name__}' (.*) did not return artifacts as a "
        "dictionary of string artifact names with their corresponding objects",
    ):
        _evaluate_custom_metric(0, non_str_artifact_name, eval_df, metrics)


@pytest.mark.parametrize(
    "fn, expectation",
    [
        (lambda eval_df, _: {"pred_sum": sum(eval_df["prediction"])}, does_not_raise()),
        (lambda eval_df, builtin_metrics: ({"test": 1.1}, {"a_list": [1, 2, 3]}), does_not_raise()),
        (
            lambda _, __: 3,
            pytest.raises(
                MlflowException,
                match="'<lambda>' (.*) did not return in an expected format",
            ),
        ),
    ],
)
def test_evaluate_custom_metric_lambda(fn, expectation):
    eval_df = pd.DataFrame({"prediction": [1.2, 1.9, 3.2], "target": [1, 2, 3]})
    metrics = _get_regressor_metrics(eval_df["target"], eval_df["prediction"])
    with expectation:
        _evaluate_custom_metric(0, fn, eval_df, metrics)


def test_evaluate_custom_metric_success():
    eval_df = pd.DataFrame({"prediction": [1.2, 1.9, 3.2], "target": [1, 2, 3]})
    metrics = _get_regressor_metrics(eval_df["target"], eval_df["prediction"])

    def example_custom_metric(_, given_metrics):
        return {
            "example_count_times_1_point_5": given_metrics["example_count"] * 1.5,
            "sum_on_label_minus_5": given_metrics["sum_on_label"] - 5,
            "example_np_metric_1": np.float32(123.2),
            "example_np_metric_2": np.ulonglong(10000000),
        }

    res_metrics, res_artifacts = _evaluate_custom_metric(0, example_custom_metric, eval_df, metrics)
    assert res_metrics == {
        "example_count_times_1_point_5": metrics["example_count"] * 1.5,
        "sum_on_label_minus_5": metrics["sum_on_label"] - 5,
        "example_np_metric_1": np.float32(123.2),
        "example_np_metric_2": np.ulonglong(10000000),
    }
    assert res_artifacts is None

    def example_custom_metric_with_artifacts(given_df, given_metrics):
        return (
            {
                "example_count_times_1_point_5": given_metrics["example_count"] * 1.5,
                "sum_on_label_minus_5": given_metrics["sum_on_label"] - 5,
                "example_np_metric_1": np.float32(123.2),
                "example_np_metric_2": np.ulonglong(10000000),
            },
            {
                "pred_target_abs_diff": np.abs(given_df["prediction"] - given_df["target"]),
                "example_dictionary_artifact": {"a": 1, "b": 2},
            },
        )

    res_metrics_2, res_artifacts_2 = _evaluate_custom_metric(
        0, example_custom_metric_with_artifacts, eval_df, metrics
    )
    assert res_metrics_2 == {
        "example_count_times_1_point_5": metrics["example_count"] * 1.5,
        "sum_on_label_minus_5": metrics["sum_on_label"] - 5,
        "example_np_metric_1": np.float32(123.2),
        "example_np_metric_2": np.ulonglong(10000000),
    }
    assert "pred_target_abs_diff" in res_artifacts_2
    assert res_artifacts_2["pred_target_abs_diff"].equals(
        np.abs(eval_df["prediction"] - eval_df["target"])
    )

    assert "example_dictionary_artifact" in res_artifacts_2
    assert res_artifacts_2["example_dictionary_artifact"] == {"a": 1, "b": 2}


def test_custom_metric(binary_logistic_regressor_model_uri, breast_cancer_dataset):
    def example_custom_metric(eval_df, given_metrics):
        return {
            "true_count": given_metrics["true_negatives"] + given_metrics["true_positives"],
            "positive_count": np.sum(eval_df["prediction"]),
        }

    with mlflow.start_run() as run:
        result = evaluate(
            binary_logistic_regressor_model_uri,
            breast_cancer_dataset._constructor_args["data"],
            model_type="classifier",
            targets=breast_cancer_dataset._constructor_args["targets"],
            dataset_name=breast_cancer_dataset.name,
            evaluators="default",
            custom_metrics=[example_custom_metric],
        )

    _, metrics, _, _ = get_run_data(run.info.run_id)

    model = mlflow.pyfunc.load_model(binary_logistic_regressor_model_uri)
    _, _, predict_fn, _ = _extract_raw_model_and_predict_fn(model)
    y = breast_cancer_dataset.labels_data
    y_pred = predict_fn(breast_cancer_dataset.features_data)

    expected_metrics = _get_classifier_per_class_metrics(y, y_pred)

    assert "true_count_on_data_breast_cancer_dataset" in metrics
    assert np.isclose(
        metrics["true_count_on_data_breast_cancer_dataset"],
        expected_metrics["true_negatives"] + expected_metrics["true_positives"],
        rtol=1e-3,
    )

    assert "true_count" in result.metrics
    assert np.isclose(
        result.metrics["true_count"],
        expected_metrics["true_negatives"] + expected_metrics["true_positives"],
        rtol=1e-3,
    )

    assert "positive_count_on_data_breast_cancer_dataset" in metrics
    assert np.isclose(
        metrics["positive_count_on_data_breast_cancer_dataset"], np.sum(y_pred), rtol=1e-3
    )

    assert "positive_count" in result.metrics
    assert np.isclose(result.metrics["positive_count"], np.sum(y_pred), rtol=1e-3)


def test_load_custom_metric_artifact_returns_given_evaluation_artifact_as_is():
    csv_artifact = CsvEvaluationArtifact("some/path")
    result = _load_custom_metric_artifact(
        "example_dataset", TempDir(), "csv_artifact", csv_artifact, 0, ""
    )
    assert csv_artifact is result


def test_load_custom_metric_artifact_raise_exception_for_non_file_artifact_path():
    with TempDir() as tmp:
        with pytest.raises(
            MlflowException,
            match=f"produced an unsupported artifact 'non_file_artifact' with path '{tmp.path()}'",
        ):
            _load_custom_metric_artifact(
                "example_dataset", tmp, "non_file_artifact", tmp.path(), 0, ""
            )


def test_load_custom_metric_artifact_raise_exception_for_unsupported_file_extension():
    with TempDir() as tmp:
        path = tmp.path("invalid_ext_example.some_ext")
        with open(path, "w") as f:
            f.write("some stuff that shouldn't be read")
        with pytest.raises(
            MlflowException,
            match=f"produced an unsupported artifact 'invalid_ext_artifact' with path '{path}'",
        ):
            _load_custom_metric_artifact(
                "example_dataset", tmp, "invalid_ext_artifact", path, 0, ""
            )


def test_load_custom_metric_artifact_for_image_files():
    import matplotlib.pyplot as plt
    from PIL.Image import open as open_image

    fig = plt.figure()
    plt.plot([1, 2, 3])
    with TempDir() as tmp:
        fig.savefig(tmp.path("test.png"))
        fig.savefig(tmp.path("test.jpg"))
        fig.savefig(tmp.path("test.jpeg"))

        with mlflow.start_run() as run:
            artifact_1 = _load_custom_metric_artifact(
                "example_dataset", tmp, "png_img_artifact", tmp.path("test.png"), 0, ""
            )
            artifact_2 = _load_custom_metric_artifact(
                "example_dataset", tmp, "jpg_img_artifact", tmp.path("test.jpg"), 0, ""
            )
            artifact_3 = _load_custom_metric_artifact(
                "example_dataset", tmp, "jpeg_img_artifact", tmp.path("test.jpeg"), 0, ""
            )

        _, _, _, artifacts = get_run_data(run.info.run_id)
        assert set(artifacts) == {
            "png_img_artifact_on_data_example_dataset.png",
            "jpg_img_artifact_on_data_example_dataset.jpg",
            "jpeg_img_artifact_on_data_example_dataset.jpeg",
        }

        assert isinstance(artifact_1, ImageEvaluationArtifact)
        assert isinstance(artifact_2, ImageEvaluationArtifact)
        assert isinstance(artifact_3, ImageEvaluationArtifact)

        assert open_image(tmp.path("test.png")) == artifact_1.content
        assert open_image(tmp.path("test.jpg")) == artifact_2.content
        assert open_image(tmp.path("test.jpeg")) == artifact_3.content


def test_load_custom_metric_artifact_for_json_file():
    with TempDir() as tmp:
        json.dump([1, 2, 3], open(tmp.path("test.json"), "w"))
        with mlflow.start_run() as run:
            artifact = _load_custom_metric_artifact(
                "example_dataset", tmp, "json_artifact", tmp.path("test.json"), 0, ""
            )
        _, _, _, artifacts = get_run_data(run.info.run_id)
        assert set(artifacts) == {"json_artifact_on_data_example_dataset.json"}
        assert isinstance(artifact, JsonEvaluationArtifact)
        assert json.load(open(tmp.path("test.json"), "r")) == [1, 2, 3]


def test_load_custom_metric_artifact_for_npy_file():
    with TempDir() as tmp:
        np_arr = np.array([1, 2, 3])
        np.save(tmp.path("test.npy"), np_arr, allow_pickle=False)
        with mlflow.start_run() as run:
            artifact = _load_custom_metric_artifact(
                "example_dataset", tmp, "npy_artifact", tmp.path("test.npy"), 0, ""
            )
        _, _, _, artifacts = get_run_data(run.info.run_id)
        assert set(artifacts) == {"npy_artifact_on_data_example_dataset.npy"}
        assert isinstance(artifact, NumpyEvaluationArtifact)
        assert np.array_equal(np.load(tmp.path("test.npy")), np_arr)


def test_load_custom_metric_artifact_for_csv_file():
    with TempDir() as tmp:
        df = pd.DataFrame({"test": [1, 2, 3]})
        df.to_csv(tmp.path("test.csv"), index=False)
        with mlflow.start_run() as run:
            artifact = _load_custom_metric_artifact(
                "example_dataset", tmp, "csv_artifact", tmp.path("test.csv"), 0, ""
            )
        _, _, _, artifacts = get_run_data(run.info.run_id)
        assert set(artifacts) == {"csv_artifact_on_data_example_dataset.csv"}
        assert isinstance(artifact, CsvEvaluationArtifact)
        assert df.equals(pd.read_csv(tmp.path("test.csv")))


def test_load_custom_metric_artifact_for_parquet_file():
    with TempDir() as tmp:
        df = pd.DataFrame({"test": [1, 2, 3]})
        df.to_parquet(tmp.path("test.parquet"))
        with mlflow.start_run() as run:
            artifact = _load_custom_metric_artifact(
                "example_dataset", tmp, "parquet_artifact", tmp.path("test.parquet"), 0, ""
            )
        _, _, _, artifacts = get_run_data(run.info.run_id)
        assert set(artifacts) == {"parquet_artifact_on_data_example_dataset.parquet"}
        assert isinstance(artifact, ParquetEvaluationArtifact)
        assert df.equals(pd.read_parquet(tmp.path("test.parquet")))


def test_load_custom_metric_artifact_for_dataframe_object():
    df = pd.DataFrame({"test": [1, 2, 3]})
    with TempDir() as tmp:
        with mlflow.start_run() as run:
            artifact = _load_custom_metric_artifact(
                "example_dataset", tmp, "df_artifact", df, 0, ""
            )
    _, _, _, artifacts = get_run_data(run.info.run_id)
    assert set(artifacts) == {"df_artifact_on_data_example_dataset.csv"}
    assert isinstance(artifact, CsvEvaluationArtifact)
    assert df.equals(artifact.content)


def test_load_custom_metric_artifact_for_ndarray_object():
    np_arr = np.array([1, 2, 3])
    with TempDir() as tmp:
        with mlflow.start_run() as run:
            artifact = _load_custom_metric_artifact(
                "example_dataset", tmp, "ndarray_artifact", np_arr, 0, ""
            )
    _, _, _, artifacts = get_run_data(run.info.run_id)
    assert set(artifacts) == {"ndarray_artifact_on_data_example_dataset.npy"}
    assert isinstance(artifact, NumpyEvaluationArtifact)
    assert np.array_equal(artifact.content, np_arr)


def test_load_custom_metric_artifact_for_plt_figure_objects():
    import matplotlib.pyplot as plt
    from PIL import Image
    import io

    fig = plt.figure()
    plt.plot([1, 2, 3])
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    with TempDir() as tmp:
        with mlflow.start_run() as run:
            artifact = _load_custom_metric_artifact(
                "example_dataset", tmp, "plt_figure_artifact", fig, 0, ""
            )
    _, _, _, artifacts = get_run_data(run.info.run_id)
    assert set(artifacts) == {"plt_figure_artifact_on_data_example_dataset.png"}
    assert isinstance(artifact, ImageEvaluationArtifact)
    assert img == artifact.content


def test_load_custom_metric_artifact_for_json_serializable_objects():
    ex_dict = {"a": 1, "b": "test", "c": 1.2}
    ex_list = [1, 2, 3, "test"]

    with TempDir() as tmp:
        with mlflow.start_run() as run:
            artifact_1 = _load_custom_metric_artifact(
                "example_dataset", tmp, "json_dict_artifact", ex_dict, 0, ""
            )
            artifact_2 = _load_custom_metric_artifact(
                "example_dataset", tmp, "json_list_artifact", ex_list, 0, ""
            )
    _, _, _, artifacts = get_run_data(run.info.run_id)
    assert set(artifacts) == {
        "json_dict_artifact_on_data_example_dataset.json",
        "json_list_artifact_on_data_example_dataset.json",
    }

    assert isinstance(artifact_1, JsonEvaluationArtifact)
    assert isinstance(artifact_2, JsonEvaluationArtifact)

    assert ex_dict == artifact_1.content
    assert ex_list == artifact_2.content


def test_load_custom_metric_artifact_raise_exception_for_unsupported_artifact_object():
    class ExampleUnsupportedObject:
        def __init__(self):
            self.x = 10

    unsupported_object = ExampleUnsupportedObject()
    with TempDir() as tmp:
        with pytest.raises(
            MlflowException,
            match=f"produced an unsupported artifact 'unsupported_artifact' with type "
            f"'{type(unsupported_object)}'",
        ):
            _load_custom_metric_artifact(
                "example_dataset", tmp, "unsupported_artifact", unsupported_object, 0, ""
            )

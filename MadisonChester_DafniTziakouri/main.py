import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, \
    StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc
from numpy import round
from sklearn.pipeline import Pipeline
# from sklearnex import patch_sklearn
import os
import matplotlib.pyplot as plt

from utils import TextPreprocessor, FeatureGenerator, remove_nan_questions

# patch_sklearn()  # to speed up scikit-learn


def basic_sol_without_pipeline(
        x_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
    print("Text preprocessing")
    text_prep = TextPreprocessor()
    x_train = text_prep.transform(x_train)

    print("Feature vectors generation from preprocessed text")
    fg = FeatureGenerator(('cv', ), ('stack', ), extra_features=tuple())
    fg.fit(x_train)
    x_train = fg.transform(x_train)

    logistic = LogisticRegression(solver="liblinear", random_state=123)
    single_split_score_assessment(logistic, x_train, y_train)
    # cross_validation_score_assessment(logistic, x_train, y_train)


def basic_sol_with_pipeline(
        x_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
    pipe = Pipeline(
        [('preprocessor', TextPreprocessor()),
         ('generator', FeatureGenerator(('cv', ), ('stack', ),
                                        extra_features=tuple())),
         ('model',
          LogisticRegression(solver="liblinear", random_state=123))])
    single_split_score_assessment(pipe, x_train, y_train)
    # cross_validation_score_assessment(pipe, x_train, y_train)


def improved_sol_with_pipeline(
        x_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
    pipe = Pipeline(
        [('preprocessor', TextPreprocessor()),
         ('generator', FeatureGenerator(
             ('tf_idf', ), ('stack', ))),
         ('model', RandomForestClassifier(random_state=123)),
         ])
    single_split_score_assessment(pipe, x_train, y_train)


def cross_validation_score_assessment(
        model, x_train, y_train, n_splits: int = 5) -> None:
    print("5-fold cross-validation training")
    _scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    results: dict = cross_validate(
        estimator=model, X=x_train, y=y_train,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123),
        scoring=_scoring, return_train_score=True, verbose=2,
        n_jobs=2 * os.cpu_count() // 3)
    print("Cross-validation results", results)


def single_split_score_assessment(model, x_train, y_train):
    x_train, x_test, y_train, y_test = train_test_split(
        x_train, y_train, test_size=0.2, random_state=123)
    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    print("TRAINING results:\n", classification_report(y_train, y_pred_train))
    print("TESTING results:\n", classification_report(y_test, y_pred_test))

    fpr_train, tpr_train, _ = roc_curve(
        y_train, model.predict_log_proba(x_train)[:, 1])
    auc_roc_train = auc(fpr_train, tpr_train)
    fpr_test, tpr_test, _ = roc_curve(
        y_test, model.predict_log_proba(x_test)[:, 1])
    auc_roc_test = auc(fpr_test, tpr_test)
    print("Training AUC:", auc_roc_train)
    print("Testing AUC:", auc_roc_test)

    plt.plot(fpr_train, tpr_train,
             label=f'Train (AUC = {round(auc_roc_train, 3)})')
    plt.plot(fpr_test, tpr_test,
             label=f'Test (AUC = {round(auc_roc_test, 3)})')
    plt.legend()
    plt.show()
    plt.close()

    # roc_curve_train = RocCurveDisplay(
    #     fpr=fpr_train, tpr=tpr_train, roc_auc=auc_roc_train,
    #     estimator_name='Train').plot()
    # roc_curve_test = RocCurveDisplay(
    #     fpr=fpr_test, tpr=tpr_test,
    #     roc_auc=auc_roc_test, estimator_name='Test').plot()
    # plt.show()


def main() -> None:
    _path_folder_quora = "~/Datasets/QuoraQuestionPairs"
    print("Loading quora_train_data.csv")
    _train_df = pd.read_csv(
        os.path.join(_path_folder_quora, "quora_train_data.csv"))
    x_train = _train_df.loc[:, ["question1", "question2"]]
    y_train = _train_df.loc[:, "is_duplicate"]

    print("Removing nans and then splitting into train/test")
    x_train, y_train = remove_nan_questions(x_train, y_train)

    # basic_sol_with_pipeline(x_train, y_train)
    # basic_sol_without_pipeline(x_train, y_train)
    # improved_sol_with_pipeline(x_train, y_train)

    print("Defining an example pipeline")
    pipe = Pipeline(
        [('preprocessor', TextPreprocessor(to_lower=True)),
         ('generator', FeatureGenerator(
             ('cv', ), ('absolute', ))),
         ('model', LogisticRegression(random_state=123, max_iter=1000)),
         ])

    print("Using the pipeline to train and test")
    single_split_score_assessment(pipe, x_train, y_train)


if __name__ == '__main__':
    main()
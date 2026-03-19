import sys
from models.school_type_predictor_model import train_models_and_get_accuracies as train_school_type
from models.seed_predictor_model import train_models_and_get_accuracies as train_seed

VALID_MODELS = ["logistic", "random_forest", "gradient_boosting", "svm"]


def print_metrics(results):
    for model_name, metrics in results.items():
        print(f"  [{model_name}]")
        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1 Score:  {metrics['f1']:.4f}")
        print(f"    Confusion Matrix:")
        for row in metrics["confusion_matrix"]:
            print(f"      {row}")
        print(f"    Classification Report:")
        for line in metrics["classification_report"].splitlines():
            print(f"      {line}")
        print()


def main():
    # If user passed model names, use them; otherwise run all
    if len(sys.argv) > 1:
        model_types = sys.argv[1:]
        invalid = [m for m in model_types if m not in VALID_MODELS]
        if invalid:
            print(f"Invalid models: {invalid}")
            print(f"Valid options: {VALID_MODELS}")
            sys.exit(1)
    else:
        model_types = VALID_MODELS # use all valid models

    print(f"Training models: {model_types}\n")

    # Question 1: Predict if a school is private or public
    print("=" * 50)
    print("SCHOOL TYPE (Public vs Private)")
    print("=" * 50)
    school_type_results = train_school_type(model_types=model_types)
    print_metrics(school_type_results)

    # Question 2: Predict which seed a team will be
    print("=" * 50)
    print("SEED PREDICTION")
    print("=" * 50)
    seed_results = train_seed(model_types=model_types)
    print_metrics(seed_results)


main()
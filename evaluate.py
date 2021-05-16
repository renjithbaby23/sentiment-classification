import numpy as np
from silence_tensorflow import silence_tensorflow

silence_tensorflow()

import tensorflow as tf
import tensorflow_text

# do not remove the above unused import

tf.get_logger().setLevel("FATAL")

# SAVED_MODEL = "./apple5_model/"
SAVED_MODEL = "../apple5_model"
CLASS_NAMES = {0: "neutral", 1: "positive", 2: "negative"}

loaded_model = tf.saved_model.load(SAVED_MODEL)


def print_my_examples(inputs: list, results: np.array) -> None:
    """pretty pring the inputs and predictions

    Args:
        inputs (list): Input texts as list of sentences or phraases
        results (np.array): results/ predictions

    Returns:
        None:
    """
    print("*" * 100)
    print("Prediction on some sample data...")
    print()
    print(
        "------------------------------------------------------------------------------------------------------"
    )
    print(
        f"input: <Input string>                 : score: [neutral  | positive  |  negative]  : class: Sentiment"
    )
    print(
        "------------------------------------------------------------------------------------------------------"
    )
    result_for_printing = [
        f"input: {inputs[i]:<30} : score: {results[i]}  : class: {CLASS_NAMES[np.argmax(results[i])]}"
        for i in range(len(inputs))
    ]
    print(*result_for_printing, sep="\n")
    print("*" * 100)


if __name__ == "__main__":

    # some dummy examples
    examples = [
        "this is such an amazing movie!",
        "The movie was great!",
        "The movie was meh.",
        "The movie was okish.",
        "The movie was terrible...",
        "how can I login to app store?",
    ]

    # predicting on the dummy examples
    predictions = loaded_model(tf.constant(examples))

    # pretty pring the results
    print_my_examples(examples, predictions)

    # deleting the model manually before exiting the program
    # to avoid a warning message shown in tf 2.4
    del loaded_model

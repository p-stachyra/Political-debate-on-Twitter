import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()
import sys

from sklearn.metrics import confusion_matrix

def assignPenalty(manual_labels, model_labels):
    """Evaluates if an inconsistency occurred. If the inconsistency is related to confusing anger
    with disgust, no penalty is assigned. It is because disgust is usually related to anger and vice
    versa.
    Expects an array containing manually assigned labels and an array of those assigned by the model."""

    y_pred = []
    y_man = []
    correct = 0
    for i in range(len(manual_labels)):
            if manual_labels[i] == model_labels[i]:
                correct += 1
                # if the same, append the original labels
                y_pred.append(model_labels[i])
                y_man.append(manual_labels[i])

            elif (manual_labels[i] in ["anger", "disgust"]) & (model_labels[i] in ["anger", "disgust"]):
                correct += 1
                # if anger confused with disgust, append one of them to both lists
                y_pred.append(model_labels[i])
                y_man.append(model_labels[i])

            else:
                # if incorrect, append the original labels
                y_pred.append(model_labels[i])
                y_man.append(manual_labels[i])

    return correct, len(manual_labels), y_pred, y_man

def plotConfusionMatrix(labels_true, labels_predicted, labels_to_include, plot_path):
    """Produces a plot of a confusion matrix and saves it to the disk.
    Expects correct labels array, predicted labels array, an array with the labels to display,
    a path to which the plot will be saved to."""

    array_cm = confusion_matrix(labels_true, labels_predicted, labels=labels_to_include)
    df_cm = pd.DataFrame(array_cm, index=[l for l in labels_to_include], columns=[l for l in labels_to_include])
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True)
    plt.savefig(plot_path)

    return 0

def generateReport(incorrect, number_preds, total_number_tweets, report_path):
    """Generates a text report for the accuracy.
    Expects the number of inconsistencies, the total number of predictions made by the model,
    total number of tweets analyzed during the study, the path to save the text report file to."""

    # generate a log file (a report with performance metrics)
    with open(report_path, "w") as fh:
        fh.write("""A fraction of correctly classified emotions: model's accuracy

                The performance metric is based on how well the model is able to predict the most dominant emotion embedded in a tweet. 
                To assess that, 50 tweets were sampled randomly and labelled manually. 
                If a manual evaluation assigns anger to a tweet and model predicted disgust, it is not counted as an error and vice versa. 
                It is because these labels are very close to each other and even a human being might have a problem to tell these two.

                Accuracy: %0.3f.
                The number of studied samples: %d.
                The fraction of all tweets: %0.3f.""" % ((incorrect / number_preds), number_preds, (number_preds / total_number_tweets)))

    return 0


def measurePerformance(path_to_csv):
    """Driver function which reads the CSV file containing labels from manual and automatic evaluation,
    calls assignPenalty function to assess the errors, generates a confusion matrix plot and generates a
    text file report.
    Expects a string: path to the CSV file."""

    if not os.path.isdir("performance_metrics"):
        os.mkdir("performance_metrics")

    try:
        # read the CSV file with assigned labels
        performance_df = pd.read_csv(path_to_csv)
        array_manual_label = performance_df["Qualitative_label"]
        array_model_label = performance_df["Model_label"]
        # measure the discrepancies
        errors, labels_n, label_pred, label_man = assignPenalty(array_manual_label, array_model_label)

        # save performance metrics
        plotConfusionMatrix(label_man,
                            label_pred,
                            ["anger", "disgust", "joy", "sadness", "surprise", "fear", "others"],
                            "performance_metrics/confusion_matrix.png")
        generateReport(errors,
                       labels_n,
                       30000,
                       "performance_metrics/Performance_Metrics.txt")

    except Exception as e:
        print("[ ! ] Error occurred. Error message:\n%s" % e)
        sys.exit(-1)

    sys.exit(0)

if __name__ == "__main__":
    measurePerformance("qualitative_evaluation_emotions/qualitatively_analysed.csv")

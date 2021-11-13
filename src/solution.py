# Python 3.8.10
from libsvm.svmutil import *
import numpy as np
import scipy.io

# file paths for loading training and testing sample data
train_data_file_path = '../data/trainData.mat'
test_data_file_path = '../data/testData.mat'


# region Helper methods

def load_data_from_file(file_path):
    assert isinstance(file_path, str)
    data = scipy.io.loadmat(file_path)
    y = np.ravel(data['Y'])
    x1 = data['X1']
    x2 = data['X2']
    x3 = data['X3']
    return y, [x1, x2, x3]


def get_parameter_str(parameters):
    assert isinstance(parameters, list)
    return " ".join(parameters)


def train_svm_model(y, x, params=''):
    assert isinstance(y, np.ndarray)
    assert len(y.shape) == 1
    assert isinstance(x, np.ndarray)
    return svm_train(y, x, params)


def predict_svm_test(y, x, m, params=''):
    assert isinstance(y, np.ndarray)
    assert len(y.shape) == 1
    assert isinstance(x, np.ndarray)
    assert isinstance(m, svm_model)
    return svm_predict(y, x, m, params)


# endregion

# region: Methods Defining Steps0

def get_accuracy(result):
    assert isinstance(result, tuple)
    assert isinstance(result[1], tuple)
    assert isinstance(result[1][0], float)
    return result[1][0]


def print_accuracies(x_results, step='Step 0', case='Case 0'):
    assert isinstance(x_results, list)
    assert all(isinstance(result, tuple) for result in x_results)
    print(f"{step}, {case} Output:")
    for index in range(len(x_results)):
        accuracy = get_accuracy(x_results[index])
        print(f"X{index + 1} Feature Test Data Accuracy: {accuracy:0.2f}%")
    print('')


def train_and_test_data(y_train, x_trains, y_test, x_tests, train_params='', test_params=''):
    assert len(x_trains) == len(x_tests)
    assert isinstance(x_trains, list)
    assert isinstance(x_tests, list)
    x_svm_models = []
    x_results = []
    for x in x_trains:
        x_svm_models.append(train_svm_model(y_train, x, train_params))
    for index in range(len(x_tests)):
        x_results.append(predict_svm_test(y_test, x_tests[index], x_svm_models[index], test_params))
    return x_results


def step0_case1(y_train, x_trains, y_test, x_tests):
    """
    Step 0: Classification by individual features
    Output: The classification accuracy for the testing set in case (1)
    Case 1: For each of the 3 features in the training set, 𝑿𝑘 (1 ≤  𝑘 ≤ 3), train a multi-class linear SVM classifier,
    i.e., ℎ𝑘 (𝐱). Get the prediction result of ℎ𝑘 (𝐱) based on the same feature 𝑿𝑘 in the testing set and compare to
    𝒀 for computing the classification accuracy
    """
    train_params = get_parameter_str(['-c', '10', '-t', '0', '-q'])
    test_params = get_parameter_str(['-q'])
    x_results = train_and_test_data(y_train, x_trains, y_test, x_tests, train_params, test_params)
    print_accuracies(x_results, 'Step0', 'Case1')
    return x_results


def step0_case2(y_train, x_trains, y_test, x_tests):
    """
    Step 0: Classification by individual features
    Output: The classification accuracy for the testing set in case (2)
    Case 2: Based on the SVM classifiers ℎ𝑘 (𝐱), we can also obtain 𝑝𝑘 (𝑤𝑖 |𝐱), the (posterior) probability of sample
    𝐱 that it belongs to the 𝑖-th category (𝑤𝑖) according to feature 𝑿𝑘 (1 ≤ 𝑘 ≤ 3). This can be done by using the
    parameter ‘-b 1’ option in training and testing (check http://www.csie.ntu.edu.tw/~cjlin/libsvm/ for more details).
    Train the SVM classifiers with this option and report the classification accuracies on the testing set based on the
    3 features respectively.
    """
    train_params = get_parameter_str(['-c', '10', '-t', '0', '-b', '1', '-q'])
    test_params = get_parameter_str(['-b', '1', '-q'])
    x_results = train_and_test_data(y_train, x_trains, y_test, x_tests, train_params, test_params)
    print_accuracies(x_results, 'Step0', 'Case2')
    return x_results


# endregion

# region: methods Defining Step1

def get_probabilities(result):
    assert isinstance(result, tuple)
    assert isinstance(result[2], list)
    return np.array(result[2])


def calculate_avg_probabilities(x_results):
    assert isinstance(x_results, list)
    assert all(isinstance(result, tuple) for result in x_results)
    x_predictions = []
    for result in x_results:
        x_predictions.append(get_probabilities(result))
    assert all(isinstance(x_prediction, np.ndarray) for x_prediction in x_predictions)
    return np.mean(x_predictions, axis=0)


def get_class_prediction(average_x_predictions):
    assert isinstance(average_x_predictions, np.ndarray)
    return average_x_predictions.argmax(axis=1) + 1


def calculate_accuracy(actual_y, predicted_y):
    assert isinstance(actual_y, np.ndarray)
    assert len(actual_y.shape) == 1
    assert isinstance(predicted_y, np.ndarray)
    assert len(predicted_y.shape) == 1
    assert len(actual_y) == len(predicted_y)
    return (np.count_nonzero(actual_y == predicted_y) / len(actual_y)) * 100


def step1(y_test, x_results):
    """
    Step 1: Feature combination by fusion of classifiers
    Output: The classification accuracy in the testing set and compare it to that of (2) in Step 0.
    Instructions: Directly combine the 3 SVM classifiers with probability output i.e., 𝑝 𝑘 (𝑤𝑖 |𝐱) (1 ≤ 𝑘 ≤ 3), in (2)
    of Step 0. Combine the 3 classifiers by probability fusion as 𝑝(𝑤𝑖 |𝐱) = ∑𝑘 𝑝𝑘 (𝑤𝑖 |𝐱) ⁄ 3. The final recognition
    result is 𝑤 𝑖∗ = argmax 𝑖 𝑝(𝑤𝑖 |𝐱).
    """
    x_avg_probabilities = calculate_avg_probabilities(x_results)
    predicted_y = get_class_prediction(x_avg_probabilities)
    x_avg_s1_accuracy = calculate_accuracy(y_test, predicted_y)
    print("Step 1 Output:")
    print(f"Test Data Accuracy using Feature Combination by Fusion of Classifiers: {x_avg_s1_accuracy:.2f}%\n")


# endregion

# region Methods Defining Step2

def get_concatenated_x(x_vector):
    assert isinstance(x_vector, tuple)
    all(isinstance(x_i, np.ndarray) for x_i in x_vector)
    return np.concatenate(x_vector, axis=1)


def step2(y_train, y_test, x_trains, x_tests, x_results):
    """
    Step 2: Feature combination by simple concatenation.
    Output: The classification accuracy in the testing set and compare it to that of (1) in Step 0.
    Instructions: Directly concatenate the 3 features 𝐗𝑘, 1 ≤ 𝑘 ≤ 3 to form a single feature, i.e.
    𝐗 = [𝐗1 , . . . , 𝐗𝐾 ]; train a linear SVM classifier based on 𝐗 and obtain the classification accuracy for the
    testing set.
    """
    concatenated_x_train = get_concatenated_x(tuple(x_trains))
    train_params = get_parameter_str(['-c', '10', '-t', '0', '-q'])
    concatenated_x_svm_model = svm_train(y_train, concatenated_x_train, train_params)
    test_params = get_parameter_str(['-q'])
    concatenated_x_test = get_concatenated_x(tuple(x_tests))
    concatenated_x_results = predict_svm_test(y_test, concatenated_x_test, concatenated_x_svm_model, test_params)
    concatenated_x_accuracy = get_accuracy(concatenated_x_results)
    print("Step 2 Output:")
    print(f"Test Data Accuracy Using Model Trained From Feature Concatenation: {concatenated_x_accuracy:.2f}%\n")


# endregion

def main():
    """
    Load testing & training data from file
    """
    y_train, x_trains = load_data_from_file(train_data_file_path)
    y_test, x_tests = load_data_from_file(test_data_file_path)

    x_results_s0_c1 = step0_case1(y_train, x_trains, y_test, x_tests)
    x_results_s0_c2 = step0_case2(y_train, x_trains, y_test, x_tests)
    step1(y_test, x_results_s0_c2)
    step2(y_train, y_test, x_trains, x_tests, x_results_s0_c1)


# Using the special variable
# __name__
if __name__ == "__main__":
    main()

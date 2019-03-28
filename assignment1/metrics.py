def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, accuracy, f1 - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
    
    tn = 0
    tp = 0
    fn = 0
    fp = 0
    for i, ground_truth_i in enumerate(ground_truth):
        if ground_truth_i == prediction[i] == True:
            tp += 1
        elif ground_truth_i == prediction[i] == False:
            tn += 1
        elif ground_truth_i == False and ground_truth_i != prediction[i]:
            fn += 1
        elif ground_truth_i == True and ground_truth_i != prediction[i]:
            fp += 1
        
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * ((precision * recall) / (precision + recall))

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    TP=0
    for i, ground_truth_i in enumerate(ground_truth):
        if ground_truth_i == prediction[i]:
            TP+=1
    # TODO: Implement computing accuracy
    return TP/len(ground_truth)

from string import punctuation, digits
import numpy as np
import random
import utils


#==============================================================================
#===  PART I  =================================================================
#==============================================================================



def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices



def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        `feature_vector` - numpy array describing the given data point.
        `label` - float, the correct classification of the data
            point.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - float representing the offset parameter.
    Returns:
        the hinge loss, as a float, associated with the given data point and
        parameters.
    """
    # Your code here
    test=(np.dot(feature_vector,theta)+theta_0)*label
    return (1-test) if test<1 else 0
    raise NotImplementedError



def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the hinge loss for given classification parameters averaged over a
    given dataset

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - real valued number representing the offset parameter.
    Returns:
        the hinge loss, as a float, associated with the given dataset and
        parameters.  This number should be the average hinge loss across all of
    """

    # Your code here
    # hinge_loss_single
    loss=0
    n=len(feature_matrix)
    for i in range(n):
        loss+=hinge_loss_single(feature_matrix[i],labels[i],theta=theta,theta_0=theta_0)
    return loss/n
    raise NotImplementedError




def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Updates the classification parameters `theta` and `theta_0` via a single
    step of the perceptron algorithm.  Returns new parameters rather than
    modifying in-place.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.
    Returns a tuple containing two values:
        the updated feature-coefficient parameter `theta` as a numpy array
        the updated offset parameter `theta_0` as a floating point number
    """
    # Your code here
    test=(np.dot(feature_vector,current_theta)+current_theta_0)*label
    if test<=0:
        return ((label*feature_vector)+current_theta,current_theta_0+label)
    else:
        return  (current_theta,current_theta_0)
    # return ((label*feature_vector)+current_theta,current_theta_0+label) if test<=0 else (current_theta,current_theta_0)
    raise NotImplementedError



def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set: we do not stop early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns a tuple containing two values:
        the feature-coefficient parameter `theta` as a numpy array
            (found after T iterations through the feature matrix)
        the offset parameter `theta_0` as a floating point number
            (found also after T iterations through the feature matrix).
    """
    # Your code here
    nsamples=feature_matrix.shape[0]
    theeta=np.zeros(feature_matrix[0].shape[0])
    theeta_0=0
    for t in range(T):
        for i in get_order(nsamples):
            # Your code here
            # print(i,feature_matrix,labels[i],theeta,theeta_0)
            theeta,theeta_0=perceptron_single_step_update(feature_matrix[i],labels[i],theeta,theeta_0)
        # print("PERCEPTRON",theeta)
    # Your code here
    return (theeta,theeta_0)
    # raise NotImplementedError



def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given dataset.  Runs `T`
    iterations through the dataset (we do not stop early) and therefore
    averages over `T` many parameter values.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: It is more difficult to keep a running average than to sum and
    divide.

    Args:
        `feature_matrix` -  A numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns a tuple containing two values:
        the average feature-coefficient parameter `theta` as a numpy array
            (averaged over T iterations through the feature matrix)
        the average offset parameter `theta_0` as a floating point number
            (averaged also over T iterations through the feature matrix).
    """
    # Your code here
    nsamples=feature_matrix.shape[0]
    theeta=np.zeros(feature_matrix[0].shape[0])
    theeta_0=0
    th=np.zeros(feature_matrix[0].shape[0])
    th_0=0
    for t in range(T):
        for i in get_order(nsamples):
            th,th_0=perceptron_single_step_update(feature_matrix[i],labels[i],th,th_0)
            theeta=theeta+th
            theeta_0=theeta_0+th_0
    # Your code here
        # print("AVG",theeta)
    theeta=theeta/(nsamples*T)
    theeta_0=theeta_0/(nsamples*T)
    return (theeta,theeta_0)


def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        theta,
        theta_0):
    """
    Updates the classification parameters `theta` and `theta_0` via a single
    step of the Pegasos algorithm.  Returns new parameters rather than
    modifying in-place.

    Args:
        `feature_vector` - A numpy array describing a single data point.
        `label` - The correct classification of the feature vector.
        `L` - The lamba value being used to update the parameters.
        `eta` - Learning rate to update parameters.
        `theta` - The old theta being used by the Pegasos
            algorithm before this update.
        `theta_0` - The old theta_0 being used by the
            Pegasos algorithm before this update.
    Returns:
        a tuple where the first element is a numpy array with the value of
        theta after the old update has completed and the second element is a
        real valued number with the value of theta_0 after the old updated has
        completed.
    """
    # Your code here
    if label*(np.dot(theta,feature_vector)+theta_0)<=1:
        theta=(1-eta*L)*theta+eta*label*feature_vector
        theta_0=theta_0+eta*label
    else:
        theta=(1-eta*L)*theta
    return (theta,theta_0)
    raise NotImplementedError



def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T iterations
    through the data set, there is no need to worry about stopping early.  For
    each update, set learning rate = 1/sqrt(t), where t is a counter for the
    number of updates performed so far (between 1 and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.  Do
    not copy paste code from previous parts.

    Args:
        `feature_matrix` - A numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        `L` - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns:
        a tuple where the first element is a numpy array with the value of the
        theta, the linear classification parameter, found after T iterations
        through the feature matrix and the second element is a real number with
        the value of the theta_0, the offset classification parameter, found
        after T iterations through the feature matrix.
    """
    # Your code here
    nsamples=feature_matrix.shape[0]
    theeta=np.zeros(feature_matrix[0].shape[0])
    theeta_0=0
    nt=1
    for t in range(T):
        for i in get_order(nsamples):
            # Your code here
            eta=1/(nt)**0.5
            nt+=1
            theeta,theeta_0=pegasos_single_step_update(feature_matrix[i],labels[i],L,eta,theeta,theeta_0)
        # print("PEGASOS",theeta)
    # Your code here
    return (theeta,theeta_0)
    raise NotImplementedError



#==============================================================================
#===  PART II  ================================================================
#==============================================================================



##  #pragma: coderesponse template
##  def decision_function(feature_vector, theta, theta_0):
##      return np.dot(theta, feature_vector) + theta_0
##  def classify_vector(feature_vector, theta, theta_0):
##      return 2*np.heaviside(decision_function(feature_vector, theta, theta_0), 0)-1
##  #pragma: coderesponse end



def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses given parameters to classify a set of
    data points.

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - real valued number representing the offset parameter.

    Returns:
        a numpy array of 1s and -1s where the kth element of the array is the
        predicted classification of the kth row of the feature matrix using the
        given theta and theta_0. If a prediction is GREATER THAN zero, it
        should be considered a positive classification.
    """
    # Your code here
    labels=np.array([]);
    for feature_vector in feature_matrix:
        test=np.dot(feature_vector,theta)+theta_0
        if test<=0:
            labels=np.append(labels,-1)
        else:
            labels=np.append(labels,1)
    return labels
    raise NotImplementedError


def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    """
    Trains a linear classifier and computes accuracy.  The classifier is
    trained on the train data.  The classifier's accuracy on the train and
    validation data is then returned.

    Args:
        `classifier` - A learning function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        `train_feature_matrix` - A numpy matrix describing the training
            data. Each row represents a single data point.
        `val_feature_matrix` - A numpy matrix describing the validation
            data. Each row represents a single data point.
        `train_labels` - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        `val_labels` - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        `kwargs` - Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns:
        a tuple in which the first element is the (scalar) accuracy of the
        trained classifier on the training data and the second element is the
        accuracy of the trained classifier on the validation data.
    """
    # Your code here
    theta,theta_0=classifier(train_feature_matrix,train_labels,**kwargs)
    result_train_labels=classify(train_feature_matrix,theta,theta_0)
    result_val_labels=classify(val_feature_matrix,theta,theta_0)
    accuracy_res_val=accuracy(result_val_labels,val_labels)
    accuracy_res_train=accuracy(result_train_labels,train_labels)
    return (accuracy_res_train,accuracy_res_val)
    raise NotImplementedError



def extract_words(text):
    """
    Helper function for `bag_of_words(...)`.
    Args:
        a string `text`.
    Returns:
        a list of lowercased words in the string, where punctuation and digits
        count as their own words.
    """
    # Your code here
    # raise NotImplementedError

    for c in punctuation + digits:
        text = text.replace(c, ' ' + c + ' ')
    return text.lower().split()



def bag_of_words(texts, remove_stopword=True):
    """
    NOTE: feel free to change this code as guided by Section 3 (e.g. remove
    stopwords, add bigrams etc.)

    Args:
        `texts` - a list of natural language strings.
    Returns:
        a dictionary that maps each word appearing in `texts` to a unique
        integer `index`.
    """
    # Your code here
    # raise NotImplementedError
    
    indices_by_word = {}  # maps word to unique index
    stopword = []
    stopword_file = open('stopwords.txt','r')
    stopword_data=stopword_file.read()
    stopword_file.close()
    # print(stopword_data.split("\n"))
    for text in texts:
        word_list = extract_words(text)
        if remove_stopword == False:
            stopword = []
        else:
            pass
            ##### Something you need to implement for Part 9
            ##### Define what stopword is
            stopword=stopword_data.split("\n")
        for word in word_list:
            if word in indices_by_word: continue
            if word in stopword: continue
            indices_by_word[word] = len(indices_by_word)
    # print(stopword)
    return indices_by_word



def extract_bow_feature_vectors(reviews, indices_by_word, binarize=True):
    """
    Args:
        `reviews` - a list of natural language strings
        `indices_by_word` - a dictionary of uniquely-indexed words.
    Returns:
        a matrix representing each review via bag-of-words features.  This
        matrix thus has shape (n, m), where n counts reviews and m counts words
        in the dictionary.
    """
    # Your code here
    feature_matrix = np.zeros([len(reviews), len(indices_by_word)], dtype=np.float64)
    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word not in indices_by_word: continue
            feature_matrix[i, indices_by_word[word]] += 1
    if binarize:
        binarize_feature_matrix=[];
        for feature_vecotr in feature_matrix:
            binarize_feature_vector=np.array([i for i in feature_vecotr])
            # print(binarize_feature_vector)
            binarize_feature_matrix.append(binarize_feature_vector)
        # pass
    #     raise NotImplementedError
    feature_matrix=np.array(binarize_feature_matrix)
    # print(binarize_feature_matrix)
    return feature_matrix



def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the fraction of predictions that are correct.
    """
    return (preds == targets).mean()

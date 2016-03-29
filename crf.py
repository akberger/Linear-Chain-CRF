import numpy as np
import math

class CRF(object):

    def __init__(self, label_codebook, feature_codebook):
        self.label_codebook = label_codebook
        self.feature_codebook = feature_codebook
        num_labels = len(self.label_codebook)
        num_features = len(self.feature_codebook)
        self.feature_parameters = np.zeros((num_labels, num_features))
        self.transition_parameters = np.zeros((num_labels, num_labels))

    def train(self, training_set, dev_set):
        """Training function

        Feel free to adjust the hyperparameters (learning rate and batch sizes)
        """
        self.train_sgd(training_set, dev_set, 0.01, 200)

    def train_sgd(self, training_set, dev_set, learning_rate, batch_size):
        """Minibatch SGF for training linear chain CRF

        This should work. But you can also implement early stopping here
        i.e. if the accuracy does not grow for a while, stop.
        """
        num_labels = len(self.label_codebook)
        num_features = len(self.feature_codebook)

        num_batches = len(training_set) / batch_size
        total_expected_feature_count = np.zeros((num_labels, num_features))
        total_expected_transition_count = np.zeros((num_labels, num_labels))
        print '\nWith all parameters = 0, the accuracy is %s' % \
                sequence_accuracy(self, dev_set)

        count = 0
        best_accuracy = 0
        accuracy = 0
        best_feature_params = None
        best_transition_params = None
        for i in range(10):
            for j in range(num_batches):
                batch = training_set[j*batch_size:(j+1)*batch_size]
                total_expected_feature_count.fill(0)
                total_expected_transition_count.fill(0)
                total_observed_feature_count, total_observed_transition_count = self.compute_observed_count(batch)
                
                for sequence in batch:
                    transition_matrices = self.compute_transition_matrices(sequence)
                    alpha_matrix = self.forward(sequence, transition_matrices)
                    beta_matrix = self.backward(sequence, transition_matrices)
                    expected_feature_count, expected_transition_count = \
                            self.compute_expected_feature_count(sequence, alpha_matrix, beta_matrix, transition_matrices)
                    total_expected_feature_count += expected_feature_count
                    total_expected_transition_count += expected_transition_count

                feature_gradient = (total_observed_feature_count - total_expected_feature_count) / len(batch)
                transition_gradient = (total_observed_transition_count - total_expected_transition_count) / len(batch)

                self.feature_parameters += learning_rate * feature_gradient
                self.transition_parameters += learning_rate * transition_gradient

                accuracy = sequence_accuracy(self, dev_set)

                #keep best sets of params
                if accuracy > best_accuracy:
                    best_feature_params = self.feature_parameters
                    best_transition_params = self.transition_parameters
                    best_accuracy = accuracy
                elif accuracy <= best_accuracy: 
                    count += 1
                    #print count

                if count >= 60: #if accuracy has gotten greater than the max accuracy 100 times it probably won't get better again
                    print 'best accuracy: ' + str(best_accuracy)
                    #make sure params are what they were for the best dev set accuracy
                    self.feature_parameters = best_feature_params
                    self.transition_parameters = best_transition_params
                    break
                

                print accuracy

                #if accuracy == 1.0:
                #    break


    def compute_transition_matrices(self, sequence):
        """Compute transition matrices (denoted as M on the slides)

        Compute transition matrix M for all time steps.

        We add one extra dummy transition matrix at time 0. 
        This one dummy transition matrix should not be used ever, but it 
        is there to make the index consistent with the slides

        The matrix for the first time step does not use transition features
        and should be a diagonal matrix.
        Returns :
            a list of transition matrices
        """
        transition_matrices = []
        num_labels = len(self.label_codebook)
        transition_matrix = np.zeros((num_labels, num_labels))
        transition_matrices.append(transition_matrix)
        for t in range(len(sequence)):
            transition_matrix = np.zeros((num_labels, num_labels))
            for i in range(num_labels):
                if t == 0:
                    #first step is diagonal
                    transition_matrix[i,i] = math.exp(sum(self.feature_parameters[i,sequence[t].feature_vector]))
                    #sum([self.feature_parameters[i,sequence[t].feature_vector]])
                else:
                    for j in range(num_labels):
                        #completely fill in all other steps
                        transition_matrix[i,j] = math.exp(self.transition_parameters[i,j] + sum(self.feature_parameters[j,sequence[t].feature_vector]))
            transition_matrices.append(transition_matrix)

        return transition_matrices

    def forward(self, sequence, transition_matrices):
        """Compute alpha matrix in the forward algorithm

        TODO: Implement this function
        """
        num_labels = len(self.label_codebook)
        alpha_matrix = np.zeros((num_labels, len(sequence) + 1))
        for t in range(len(sequence) + 1):
            if t == 0:
                alpha_matrix[:,t] = 1
            else:
                alpha_matrix[:,t] = np.dot(alpha_matrix[:,t-1], transition_matrices[t])

        return alpha_matrix          

    def backward(self, sequence, transition_matrices):
        """Compute beta matrix in the backward algorithm

        TODO: Implement this function
        """
        num_labels = len(self.label_codebook)
        beta_matrix = np.zeros((num_labels, len(sequence) + 1))
        time = range(len(sequence) + 1)
        time.reverse()
        for t in time:
            if t == len(sequence):
                beta_matrix[:,t] = 1
            else:
                # based on the values of beta matrix and transition matrix at prev timestep
                beta_matrix[:,t] = np.dot(beta_matrix[:,t+1], transition_matrices[t+1])
        return beta_matrix

    def decode(self, sequence):
        """Find the best label sequence from the feature sequence

        TODO: Implement this function

        Returns :
            a list of label indices (the same length as the sequence)
        """
        transition_matrices = self.compute_transition_matrices(sequence)
        decode_matrix = np.zeros((len(self.label_codebook), len(sequence) + 1))
        path = [['' for i in range(len(sequence) + 1)] for j in range(len(self.label_codebook))]

        for t in range(len(sequence) + 1):
            if t == 0:
                decode_matrix[:,t] = 1
            else:
                for i in range(len(self.label_codebook)):
                    decode_matrix[i,t] = np.max(decode_matrix[:,t-1] * transition_matrices[t][:,i])
                    argmax = np.argmax(decode_matrix[:,t-1] * transition_matrices[t][:,i])
                    path[i][t] = path[argmax][t-1] + str(i)

        best_row = np.argmax(decode_matrix[:,-1])
        best_col = len(sequence)
        best_path = [int(c) for c in path[best_row][best_col]]
        return best_path

    def compute_observed_count(self, sequences):
        """Compute observed counts of features from the minibatch

        This is implemented for you.

        Returns :
            A tuple of
                a matrix of feature counts 
                a matrix of transition-based feature counts
        """
        num_labels = len(self.label_codebook)
        num_features = len(self.feature_codebook)

        feature_count = np.zeros((num_labels, num_features))
        transition_count = np.zeros((num_labels, num_labels))
        for sequence in sequences:
            for t in range(len(sequence)):
                if t > 0:
                    transition_count[sequence[t-1].label_index, sequence[t].label_index] += 1
                feature_count[sequence[t].label_index, sequence[t].feature_vector] += 1
        return feature_count, transition_count

    def compute_expected_feature_count(self, sequence, 
            alpha_matrix, beta_matrix, transition_matrices):
        """Compute expected counts of features from the sequence

        This is implemented for you.

        Returns :
            A tuple of
                a matrix of feature counts 
                a matrix of transition-based feature counts
        """
        num_labels = len(self.label_codebook)
        num_features = len(self.feature_codebook)

        feature_count = np.zeros((num_labels, num_features))
        sequence_length = len(sequence)
        Z = np.sum(alpha_matrix[:,-1])

        #gamma = alpha_matrix * beta_matrix / Z 
        gamma = np.exp(np.log(alpha_matrix) + np.log(beta_matrix) - np.log(Z))
        for t in range(sequence_length):
            for j in range(num_labels):
                feature_count[j, sequence[t].feature_vector] += gamma[j, t]

        transition_count = np.zeros((num_labels, num_labels))
        for t in range(sequence_length - 1):
            transition_count += (transition_matrices[t+1] * np.outer(alpha_matrix[:, t], beta_matrix[:,t+1])) / Z
        return feature_count, transition_count

def sequence_accuracy(sequence_tagger, test_set):
    correct = 0.0
    total = 0.0
    for sequence in test_set:
        decoded = sequence_tagger.decode(sequence)
        assert(len(decoded) == len(sequence))
        total += len(decoded)
        for i, instance in enumerate(sequence):
            if instance.label_index == decoded[i]:
                correct += 1
    return correct / total



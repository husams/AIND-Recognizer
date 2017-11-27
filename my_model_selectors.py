import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        best_score = float("inf")
        best_model = None

        for num_states in range(self.min_n_components, self.max_n_components):
            try:
                
                # Train HMM model
                hmm_model = self.base_model(num_states)

                if hmm_model is not None:
                    # Calculate BIC score
                    logL      = hmm_model.score(self.X, self.lengths)
                    p         = num_states * (self.X.shape[1] * 2 + 1)
                    logN      = np.log(len(self.X))
                    BIC_score = -2 * logL + p * logN

                    if best_score > BIC_score:
                        best_score = BIC_score
                        best_model = hmm_model
            except:
                pass
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(l)og(P(X(all but i)
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        #self.verbose =True
        best_model = None
        best_score = float('-inf')
        M          = len(self.words)

        for num_states in range(self.min_n_components, self.max_n_components):
            try:
                hmm_model = self.base_model(num_states)
                if hmm_model is not None:
                    score  = hmm_model.score(self.X, self.lengths)
                    scores = 0.0
                    for word in filter(lambda w : w != self.this_word, self.words):
                        X, lengths = self.hwords[word]
                        scores    += hmm_model.score(X, lengths)
                        
                    dic_score = score - scores / (M - 1)
                    if best_score < dic_score:
                        # Swap best model
                        best_model = dic_score
                        best_model = hmm_model
                        if self.verbose:
                            print("Best score : {}, for model with {} states".format(best_score, num_states))
            except:
                pass
        return best_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_score      = float('-inf')
        best_model      = None 
        splits          = min(3, len(self.sequences))
        split_methods   = KFold(n_splits = 3, shuffle = False, random_state = None)

        #print("Start .... ")
        for num_states in range(self.min_n_components, self.max_n_components):
            scores = []
            try:
                # We need enough data to split.
                if len(self.sequences) > 2:
                    for train, test in split_methods.split(self.sequences): 
                        self.X, self.lengths   = combine_sequences(train, self.sequences)
                        X_test,  test_lengths  = combine_sequences(test, self.sequences)

                        hmm_model = self.base_model(num_states)
                        if hmm_model is not None:
                            #print("Compute score") 
                            score  = hmm_model.score(X_test, test_lengths)
                            #print("Score : {}".format(score))
                            scores.append(score)    
                            #print("Scores : {}".format(len(scores)))

                            mean_score = np.mean(scores)
                            #print("Mean score so far is {}".format(mean))
                            if best_score < mean_score:
                                #print("1. Best score : {}".format(best_score))
                                best_score = mean_score
                                best_model = hmm_model
                                #print("2. Best score : {}".format(best_score))
                else:
                    hmm_model = self.base_model(num_states)
                    if hmm_model is not None:
                        score  = hmm_model.score(self.X, self.lengths)
                        if best_score < score:
                            best_score = score
                            best_model = hmm_model


            except:
                pass
        #print("End ... {}".format(best_model is not None))
        return best_model
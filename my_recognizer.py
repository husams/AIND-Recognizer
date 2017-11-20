import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    def compute_log_likelihood(model, X, lengths):
      try:
        return model.score(X, lengths)
      except:
        return float('-inf')

    probabilities = []
    guesses       = []

    for idx in range(0, len(test_set.get_all_Xlengths())):
        X, lengths     = test_set.get_item_Xlengths(idx)
        log_likelihood = { word : compute_log_likelihood(model, X, lengths) for word, model in models.items() }
        guess          = max(log_likelihood, key=log_likelihood.get)

        probabilities.append(log_likelihood)
        guesses.append(guess)

    return probabilities, guesses
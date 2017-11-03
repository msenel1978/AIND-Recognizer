import math
import warnings
import logging
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
    # The source code is adapted from https://github.com/sumitbinnani/Sign-Language-Recognizer
    # and https://github.com/diegoalejogm/AI-Nanodegree/tree/master/sign-language-recognizer

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    
    # Implement the recognizer
    for item in range(test_set.num_items):
        
      best_prob, best_word = -math.inf, None
      tmp_word_probabilities = {}
        
      # Get current test set sequences
      tmp_sequences, tmp_lengths = test_set.get_item_Xlengths(item)
        
      for word, model in models.items():
        try:

          # Test scores
          tmp_word_probabilities[word] = model.score(tmp_sequences, tmp_lengths)
            
        except Exception as e:
          logging.info("Exception:", e)
          tmp_word_probabilities[word] = -math.inf
            
        # Best probability so far?
        if(tmp_word_probabilities[word] > best_prob):
          best_prob, best_word = tmp_word_probabilities[word], word

      probabilities.append(tmp_word_probabilities)
      guesses.append(best_word)

    return probabilities , guesses

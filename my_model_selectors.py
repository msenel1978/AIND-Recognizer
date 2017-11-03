import math
import statistics
import warnings
import logging

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
    # The source code is adapted from https://github.com/sumitbinnani/Sign-Language-Recognizer
    # and https://github.com/diegoalejogm/AI-Nanodegree/tree/master/sign-language-recognizer

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implement model selection based on BIC scores

        # Initialization of the parameters for tracking
        # best score and model
        best_score , best_model = math.inf , None

        # Iterate over states (n_components)
        for n_components in range(self.min_n_components, self.max_n_components+1):
            try:
                # GaussianHMM with n_compotents states
                tmp_model = self.base_model(n_components)

                # BIC = -2 * logL + p * logN
                logL = tmp_model.score(self.X, self.lengths)
                n_features = self.X.shape[1]

                # p value in BIC
                n_params = n_components * (n_components - 1) + 2 * n_features * n_components

                logN = np.log(self.X.shape[0])

                # Calculate BIC score
                bic_score = -2 * logL + n_params * logN

                # Best BIC score so far
                if bic_score < best_score:
                    best_score, best_model = bic_score, tmp_model

            except Exception as e:
                logging.info("Exception:",self.this_word, e)
                continue

        return best_model if best_model is not None else self.base_model(self.n_constant)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    # The source code is adapted from https://github.com/sumitbinnani/Sign-Language-Recognizer
    # and https://github.com/diegoalejogm/AI-Nanodegree/tree/master/sign-language-recognizer

    _logL_values = None

    def log_likelihood_terms(self):

        values = {}

        for n_components in range(self.min_n_components, self.max_n_components+1):

            # Calculate logL for all words with n_components
            n_components_logL = {}
            for word in self.words.keys():
                X_word, lengths_word = self.hwords[word]
                try:
                    # GaussianHMM with n_compotents states
                    model = GaussianHMM(n_components=n_components, n_iter=1000).fit(X_word, lengths_word)

                    # Log-Likelihood Term
                    n_components_logL[word] = model.score(X_word, lengths_word)

                except Exception as e:
                    logging.info("Exception:", e)
                    continue

            values[n_components] = n_components_logL

        return (values)


    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # Implement model selection based on DIC scores


        # Initialization of the parameters for tracking
        best_score, best_n_components = -math.inf, None

        # log Likehood values inside SUM(log(P(X(all but i)
        if(self._logL_values == None):
            self._logL_values = self.log_likelihood_terms()

        logL_values = self._logL_values
        for n_components in range(self.min_n_components, self.max_n_components+1):
            
            # Calculate logL for all words with n_components
            data_likehood = logL_values[n_components]

            # Skip current n_components if self.this_word doesn't have a valid model
            if(self.this_word not in data_likehood):
                logging.info("{} does not have a valid model",format(self.this_word))
                continue
            
            # Avg. of Anti-likelihood terms (i != j)
            avg = np.mean([data_likehood[word] for word in data_likehood.keys() if word != self.this_word])

            # DIC Score
            DIC = data_likehood[self.this_word] - avg
            
            # Is this the best score so far?
            if(best_score is None or DIC > best_score):
                best_score, best_n_components = DIC, n_components

        # If could not find a best_score
        if(best_score == None):
            best_n_components = self.n_constant

        return self.base_model(best_n_components)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    # The source code is adapted from https://github.com/sumitbinnani/Sign-Language-Recognizer
    # and https://github.com/diegoalejogm/AI-Nanodegree/tree/master/sign-language-recognizer

    n_splits = 3

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implement model selection using CV

        # Initialization of the parameters for tracking
        # best score and model
        best_score , best_model = -math.inf , None

        # Iterate over states (n_components)
        for n_components in range(self.min_n_components, self.max_n_components+1):
            # Init in the loop
            tmp_scores = []
            tmp_model = None
            logL = None

            if len(self.sequences) >= self.n_splits:
                # K-Folds cross-validator
                split_method = KFold(random_state=self.random_state, n_splits=self.n_splits)

                for train, test in split_method.split(self.sequences):
                    X_train , lengths_train = combine_sequences(train, self.sequences)
                    X_test , lengths_test = combine_sequences(test, self.sequences)

                    try:
                        # Hidden Markov Model with Gaussian emissions.
                        tmp_model = GaussianHMM(n_components=n_components,
                                                covariance_type="diag", n_iter=1000,
                                                verbose=False).fit(X_train, lengths_train)

                        # Compute the log probability under the model.
                        logL = tmp_model.score(X_test, lengths_test)

                        tmp_scores.append(logL)

                    except Exception as e:
                        logging.info("Exception:",self.this_word, e)
                        break

                    # average log Likelihood score
                    avg_score = np.average(tmp_scores) if len(tmp_scores) > 0 else (-math.inf)

                    # Is this the best score so far?
                    if avg_score > best_score:
                        best_score, best_model = avg_score, tmp_model

            else:   # len(self.sequences) < self.n_splits
                logging.info('Number of sequences less than splits\n')
                try:
                    # GaussianHMM with n_compotents states
                    tmp_model = self.base_model(n_components)

                    # Compute the log probability under the model.
                    logL = tmp_model.score(self.X, self.lengths)

                    # Is this the best score so far?
                    if logL > best_score:
                        best_score, best_model = logL, tmp_model

                except Exception as e:
                        logging.info("Exception:",self.this_word, e)
                        break


        return best_model if best_model is not None else self.base_model(self.n_constant)

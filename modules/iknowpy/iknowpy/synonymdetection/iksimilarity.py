from gensim.models import fasttext as ft
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import FastTextKeyedVectors as FTKV
from gensim.models.keyedvectors import Word2VecKeyedVectors as W2VKV
import iknowpy
import os


class IKSimilarityTools(object):
    
    def __init__(self, pmodel_name, wv):
        if pmodel_name[-4:] != '.bin':
                pmodel_name = pmodel_name + '.bin'
        self.model_name = pmodel_name
        self.wordvectors = wv

    
    def most_similar(self, term, num_similar=5):
        """ Finds the num_similar most similar words to the supplied word argument. If a 
        term is multiple words, input as a string with each word separated by spaces. 
        For example "acute pulmonary hypertension"

        This method will check if a tokenized version of the input term is in the dictionary.
        If it is, it will use that to evaluate synonyms, otherwise it will provide a list of
        the words in the term and find words that correspond with that list.

        Parameters
        -----------
        term (str) - The term (word, entity, etc) for which we want to find the most similar matches.
        
        num_similar (int, optional) - The number of top similar words to be found (e.g. 5 returns the 5 most similar words 
        to the specified word based on the current model). Default 5


        Returns
        -----------
        A list of the top words

        Throws
        -----------
        KeyError - if term is not found in the vocabulary
        """
        if term.replace(" ", "_") in self.wordvectors.vocab:
            gensimpairs = self.wordvectors.most_similar(term.replace(" ", "_"), topn=num_similar)
        else:
            try:
                positives = term.split()
                gensimpairs = self.wordvectors.most_similar(positive=positives, topn=num_similar)
            except KeyError as err:
                # KeyError raised when term (fastText: and all ngrams) are NOT found in the vector space
                raise KeyError("{} is not in the vocabulary and a vector could not be found/computed".format(term)) from err

        return [pair[0].replace("_", " ") for pair in gensimpairs]
    

    def get_similarity(self, term1, term2):
        """ Gets cosine similarity of word1 and word2

        Parameters
        -----------
        term1, term2 (str) - The two terms for which one wishes to calculate cosine similarity.
        If using a tokenized model, these can be multi-word phrases (which will be tokenized by 
        this method), otherwise must be a word.

        Returns
        -----------
        Float [0,1] where 0 = not similar and 1 = identical

        Throws
        -----------
        KeyError: if term1 and/or term2 not in the model's vocabulary
        """
        try:
            term1 = term1.replace(" ", "_")
            term2 = term2.replace(" ", "_")
            sim = self.wordvectors.similarity(term1, term2)
            return sim
        except KeyError as err:
            raise KeyError("{} is not in the vocabulary, cosine similarity could not be computed".format(err.args[0])) from err
    

    def evaluate_word_pairs(self, word_pairs, delimiter, case_insensitive):
        """ Calls gensim's evaluate_word_pairs method and returns the Spearman coefficient 
        and out-of-vocabulary (OOV) ratio.

        Parameters
        -----------
        word_pairs (str) - Path to file where each line has three values: the first two 
        are the words in the word pair and the third is the human-assigned similarity rating

        delimiter (str, optional) - The character that delimits the three values in each line (default = \t)

        case_insensitive (bool, optional) - If True, convert all tokens to uppercase to before evaluation


        Returns
        -----------
        Spearman coefficient (float) - The Spearman coefficient between human similarity judgments of the word pairs
        and the model assigned scores
        
        OOV ratio (float) - The ratio of words that were out of vocabulary

        NOTE: This is to give some quantitative sense of the efficacy of the model. 
        """
        score = self.wordvectors.evaluate_word_pairs(pairs=word_pairs, delimiter=delimiter, case_insensitive=case_insensitive)
        return (score[1][0], score[2])
    

    def synonym_dict_from_string(self, source_text, use_iknow_entities=True, num_similar=5):
        """ Uses currently loaded model to determine a dictionary of synonyms for each word or
        entity in a provided string.

        Parameters
        --------------
        source_text (str) - A free string text source
        
        use_iknow_entities (bool) - whether to find synonyms for iKnow entities in the string (as opposed to words)

        num_similar (int) - Number of similar words that will be returned for each term in the source text (if exist).
        Higher num_similar ~ less strict similarity, lower num_similar ~ more strict similarity


        Returns
        --------------
        A dictionary of synonyms for each entity or word in the source

        NOTE: Right now, using iKnow entities will only check for synoyms of the iKnow entities, not for 
        their individual components. So it is one or the other
        """
        dictionary = {}
        if use_iknow_entities:
            # index the source with iknow entities
            engine = iknowpy.iKnowEngine()
            engine.index(source_text, 'en')
            # Populate dictionary with keys for each term, all with empty list for value
            for s in engine.m_index['sentences']:
                for e in s['entities']:
                    if (e['type'] in ('PathRelevant', 'NonRelevant')) or (e['index'] in dictionary):
                        continue
                    else:
                        try:
                            dictionary[e['index']] = [self.most_similar(e['index'], num_similar=num_similar)] \
                                if num_similar == 1 else self.most_similar(e['index'], num_similar=num_similar)
                        except KeyError:
                            continue
        else:
            # use words instead of entities
            words = source_text.split(' ')
            for word in words:
                if word in dictionary: continue
                else:
                    try:
                        dictionary[word] = [self.most_similar(word, num_similar=num_similar)] \
                            if num_similar == 1 else self.most_similar(word, num_similar=num_similar)
                    except KeyError:
                        continue

        return dictionary


    def synonym_dict_from_file(self, source_text, use_iknow_entities=True, num_similar=5):
        """ Uses currently loaded model to determine a dictionary of synonyms for each word or
        entity in a provided text file.

        Parameters
        --------------
        source_text (str) - The path to a file containing the source text
        
        use_iknow_entities (bool) - whether to find synonyms for iKnow entities (as opposed to words)

        num_similar (int) - Number of similar words that will be returned for each term in the source text (if exist).
        Higher num_similar ~ less strict similarity, lower num_similar ~ more strict similarity


        Returns
        --------------
        a dictionary of synonyms for each entity or word in the source

        NOTE: Right now, using iKnow entities will only check for synoyms of the iKnow entities, not for 
        their individual components. So it is one or the other.
        """
        dictionary = {}
        if use_iknow_entities:
            # index the source with iknow entities
            engine = iknowpy.iKnowEngine()
            for line in open(source_text , 'r'):
                engine.index(line, 'en')
                # Populate dictionary with keys for each term, all with empty list for value
                for s in engine.m_index['sentences']:
                    for e in s['entities']:
                        if (e['type'] in ('PathRelevant', 'NonRelevant')) or (e['index'] in dictionary):
                            continue
                        else:
                            try:
                                dictionary[e['index']] = [self.most_similar(e['index'], num_similar=num_similar)] \
                                    if num_similar == 1 else self.most_similar(e['index'], num_similar=num_similar)
                            except KeyError:
                                continue
        else:
            # use words instead of entities
            for line in open(source_text, 'r'):
                words = line.split(' ')
                for word in words:
                    if word in dictionary: continue
                    else:
                        try:
                            dictionary[word] = [self.most_similar(word, num_similar=num_similar)] \
                                if num_similar == 1 else self.most_similar(word, num_similar=num_similar)
                        except KeyError:
                            continue
        return dictionary



class IKFastTextTools(IKSimilarityTools):
    """ Subclass of IKSimilarityTools

        Contains methods to access exisiting FastText models and evaluate similarity between terms.

        Methods:
        load_vectors(self, pmodel_name): Load into memory the vectors of a previously trained model.

        Inherited methods:
        most_similar(self, term, num_similar=5): Return num_similar most similar terms to term in the model

        get_similarity(self, term1, term2): Compute cosine similarity of term1 and term2

        evaluate_word_pairs(self, word_pairs, delimiter, case_insensitive): Compute Spearman coeff. and 
        out-of-vocab ratio for a model

        synonym_dict_from_string(self, source_text, use_iknow_entities=True, num_similar=5):
        Get a dictionary of synonyms, where each key is a term in the source_text string and each
        entry is the top num_similar most similar words to that key in the model. 

        synonym_dict_from_file(self, source_text, use_iknow_entities=True, num_similar=5)
        Get a dictionary of synonyms, where each key is a term in the source_text FILE and each
        entry is the top num_similar most similar words to that key in the model. 
    """

    __PATH_PREFIX__ = os.path.join('models', 'fasttext')

    def __init__(self, pmodel_name, installdir=''): 

        # installdir will be for calls from within IRIS and is {INSTALLDIR}/mgr/python. IRIS
        # Python runtime doesn't handle relative paths well currently. 
        # Otherwise, __PATH_PREFIX__ alone is sufficient.
        if installdir != '': self.__PATH_PREFIX__ = installdir + self.__PATH_PREFIX__

        try:
            if pmodel_name[-4:] != '.bin':
                pmodel_name = pmodel_name + '.bin'
            self.wordvectors = ft.load_facebook_vectors(os.path.join(self.__PATH_PREFIX__, pmodel_name))
        except FileNotFoundError as err:
            raise FileNotFoundError('No model found with name %s' % pmodel_name) from err

        self.model_name = pmodel_name


    def load_vectors(self, pmodel_name):
        """ Loads the VECTORS of an already trained model. It is much quicker and 
        less cumbersome to use just vectors than to use the model itself, but
        still comes with the various important syntactic/semantic tools.

        Parameters
        -----------
        pmodel_name (str) - Name of the model to load vectors from

        Throws
        -----------
        FileNotFoundError - If specified model is not found.
        """
        try:
            if pmodel_name[-4:] != '.bin':
                pmodel_name = pmodel_name + '.bin'
            self.wordvectors = ft.load_facebook_vectors(os.path.join(self.__PATH_PREFIX__, pmodel_name))
        except FileNotFoundError as err:
            raise FileNotFoundError("Model with name {new} not found. Continuing use of vectors for currently loaded model ({old})".format(new=pmodel_name[:-4], 
        old=self.model_name[:-4])) from err



class IKWord2VecTools(IKSimilarityTools):
    """ Subclass of IKSimilarityTools

        Contains methods to access existing Word2Vec models and evaluate similarity between terms.

        Methods:
        load_vectors(self, pmodel_name): Load into memory the vectors of a previously trained model.

        Inherited methods:
        most_similar(self, term, num_similar=5): Return num_similar most similar terms to term in the model

        get_similarity(self, term1, term2): Compute cosine similarity of term1 and term2

        evaluate_word_pairs(self, word_pairs, delimiter, case_insensitive): Compute Spearman coeff. and 
        out-of-vocab ratio for a model

        synonym_dict_from_string(self, source_text, use_iknow_entities=True, num_similar=5):
        Get a dictionary of synonyms, where each key is a term in the source_text string and each
        entry is the top num_similar most similar words to that key in the model. 

        synonym_dict_from_file(self, source_text, use_iknow_entities=True, num_similar=5)
        Get a dictionary of synonyms, where each key is a term in the source_text FILE and each
        entry is the top num_similar most similar words to that key in the model. 
    """
    __PATH_PREFIX__ = os.path.join('models', 'word2vec', 'vectors')

    def __init__(self, pmodel_name='IKDefaultModel', installdir=''):
        
        # installdir will be for calls from within IRIS and is {INSTALLDIR}/mgr/python. IRIS
        # Python runtime doesn't handle relative paths well currently. 
        # Otherwise, __PATH_PREFIX__ alone is sufficient.
        if installdir != '': self.__PATH_PREFIX__ = installdir + self.__PATH_PREFIX__

        try:
            if pmodel_name[-4:] != '.bin':
                pmodel_name = pmodel_name + '.bin'
            self.wordvectors = W2VKV.load_word2vec_format(os.path.join(self.__PATH_PREFIX__, pmodel_name), binary=True)
        except FileNotFoundError as err:
            raise FileNotFoundError('No model found with name {}'.format(pmodel_name[:-4])) from err

        self.model_name = pmodel_name # Keeps track of what model is at use


    def load_vectors(self, pmodel_name):
        """ Loads the VECTORS of an already trained model. It is much quicker and 
        less cumbersome to use just vectors than to use the model itself, but
        still comes with the various important syntactic/semantic tools.

        Parameters
        -----------
        pmodel_name (str) - Name of the model to load vectors from

        Throws
        -----------
        FileNotFoundError - if specified model (pmodel_name) is not found
        """

        try:
            if pmodel_name[-4:] != '.bin':
                pmodel_name = pmodel_name + '.bin'
            self.wordvectors = W2VKV.load_word2vec_format(os.path.join(self.__PATH_PREFIX__, pmodel_name), binary=True)
        except FileNotFoundError as err:
            raise FileNotFoundError("Model with name {new} not found. Continuing use of vectors for currently loaded model ({old})".format(new=pmodel_name[:-4], 
        old=self.model_name[:-4])) from err




class IKSimilarityModeling():

    @classmethod
    def create_new_model(cls, corpus_path, model, epochs=5, use_iknow_entities=True, tokenize_concepts=True):
        print('Building vocabulary...\n')
        # build vocabulary
        try:
            corpus_path = SentenceIterator(corpus_path=corpus_path, use_iknow_entities=use_iknow_entities, tokenize_concepts=tokenize_concepts)
            model.build_vocab(sentences=corpus_path)
        except FileNotFoundError as err:
            raise FileNotFoundError('No corpus found at %s' % corpus_path) from err
        except RuntimeError as err:
            raise RuntimeError("Model could not be trained. If you are attempting to continue training an exisiting model, call update_model()") from err
        
        print('Finished building vocabulary.\n')
        print('Training model...\n')
        # train the model
        model.train(
            sentences=corpus_path, epochs=epochs, total_examples=model.corpus_count
        )

        print('Finished training model.\n')

    @classmethod
    def update_model(cls, corpus_path, model, use_iknow_entities, tokenize_concepts):
        # Must update the vocabulary of unique words in the corpus prior to training
        # Note that you MUST pass in update=True to not destroy existing form of the model
        sentences = SentenceIterator(corpus_path, use_iknow_entities=use_iknow_entities, tokenize_concepts=tokenize_concepts)

        model.build_vocab(sentences=sentences, update=True)
        
        model.train(
            sentences=sentences, total_examples=model.corpus_count, 
            epochs=model.epochs
        )

        
class IKFastTextModeling(IKSimilarityModeling):
    """ Subclass of IKSimilarityModeling

        Contains methods to train (create new) and retrain (update existing) models.

        Methods (class methods):
        create_new_model(cls, corpus_path, pmodel_name, epochs=5, pmin_count=10, psize=150, installdir=''):
        Creates a new model pmodel_name, trained on corpus found at corpus_path. Makes epochs number of passes
        over the corpus. Words in the corpus must appear pmin_count times to be considered. Vectors will be of 
        dimension psize. installdir is passed from IRIS Python Runtime to aid Python in locating relevant directories
        of the module on an IRIS instance.

        update_model(cls, corpus_path, pmodel_name, use_iknow_entities=True, tokenize_concepts=True, installdir=''):
        Updates the model pmodel_name, training on corpus found at corpus_path. If use_iknow_entities, will use 
        iKnow entities in training. If tokenize_concepts, will turn iKnow entities into singular tokens, joined by underscores
        (_). installdir is passed from IRIS Python Runtime to aid Python in locating relevant directories
        of the module on an IRIS instance.

        NOTE: Update is currently non-functional.
    """

    __PATH_PREFIX__ = os.path.join('models', 'fasttext')


    @classmethod
    def create_new_model(cls, corpus_path, pmodel_name, epochs=5, pmin_count=10, psize=150, installdir=''):
        """ Creates and trains (and optionally saves) a model using gensim's implementation 
        of the fastText algorithm, and then loads the KeyedVectors associated with that model.
        
        For CREATION/first time training only. To continue training an already existing
        model, use update_model().

        Parameters
        -----------
        corpus_path (str) - path to the corpus you wish to train the model with
        
        pmodel_name (str) - the name to be assigned to the model when saved. Must be unique
        or error will be raised to avoid overwriting an existing model

        epochs (int, optional) - Number of times to iterate over training corpus during training

        pmin_count (int, optional) - Minimum frequency for a word to be used in training

        psize (int, optional) - Size of vectors for training

        Returns:
        -----------
        True if model created/trained, False if could not be created

        Throws
        -----------
        FileNotFoundError - If corpus_path not found
        RuntimeError - If training an already existing model that makes it past first if statement. This
        is because build_vocab raises RuntimeError if building existing vocab without update=True (see update_model)
        """
        if installdir != '':
            model_path = installdir + IKFastTextModeling.__PATH_PREFIX__

        if pmodel_name[-4:] != '.bin':
                pmodel_name = pmodel_name + '.bin'

        if os.path.exists(os.path.join(model_path, pmodel_name)):
            raise FileExistsError("Model named {} already exists, model could not be created".format(pmodel_name[:-4]))
        
        model = ft.FastText(size=psize, sg=1, min_count=pmin_count)

        super().create_new_model(corpus_path, model, epochs)

        ft.save_facebook_model(model, path=os.path.join(model_path, pmodel_name))
        return True


    @classmethod
    def update_model(cls, corpus_path, pmodel_name, use_iknow_entities=True, tokenize_concepts=True, installdir=''):
        """ Updates an already existing model by continuing its training
        on a new corpus.

        Parameters
        -----------
        corpus_path (str) - path to the corpus being used to update the model
        
        pmodel_name (str, optional) - The name of the model to be updated, defaults to the
        model currently in use

        Return
        -----------
        True if model was updated, else False

        Throws
        -----------
        FileNotFoundError - if corpus or model not found
        """

        model_path = installdir + IKFastTextModeling.__PATH_PREFIX__

        try:
            if pmodel_name[-4:] != '.bin':
                pmodel_name = pmodel_name + '.bin'
            path = os.path.join(model_path, pmodel_name)
            model = ft.load_facebook_model(path)

            super().update_model(corpus_path, model, use_iknow_entities, tokenize_concepts)

            # Clear current contents of folders storing model and KeyedVectors files as gensim doesn't do it
            os.remove(path)
            
            ft.save_facebook_model(model, path=path)
            
        except FileNotFoundError as err:
            raise FileNotFoundError("Model could not be updated, check specified corpus and model names") from err



class IKWord2VecModeling(IKSimilarityModeling):
    """ Subclass of IKSimilarityModeling

        Contains methods to train (create new) and retrain (update existing) models.

        Methods (class methods):
        create_new_model(cls, corpus_path, pmodel_name, updateable=True, epochs=5, pmin_count=5, psize=300, installdir=''):
        Creates a new model pmodel_name, trained on corpus found at corpus_path. If updateable, will save the entire model
        to be reloaded for continued training (updating) at a later time, otherwise only save vectors. Makes epochs number of passes
        over the corpus. Words in the corpus must appear pmin_count times to be considered. Vectors will be of 
        dimension psize. installdir is passed from IRIS Python Runtime to aid Python in locating relevant directories
        of the module on an IRIS instance.

        update_model(cls, corpus_path, pmodel_name, use_iknow_entities=True, tokenize_concepts=True, installdir=''):
        Updates the model pmodel_name, training on corpus found at corpus_path. If use_iknow_entities, will use 
        iKnow entities in training. If tokenize_concepts, will turn iKnow entities into singular tokens, joined by underscores
        (_). installdir is passed from IRIS Python Runtime to aid Python in locating relevant directories
        of the module on an IRIS instance.
    """
    
    __MODEL_PATH_PREFIX__ = os.path.join('models', 'word2vec', 'trained_models')
    __VECTOR_PATH_PREFIX__ = os.path.join('models', 'word2vec', 'vectors')

    
    @classmethod
    def create_new_model(cls, corpus_path, pmodel_name, updateable=True, epochs=5, pmin_count=5, psize=150, installdir=''):
        """ Creates and trains (and optionally saves) a model using gensim's implementation 
        of the fastText algorithm, and then loads the KeyedVectors associated with that model.
        
        For CREATION/first time training only. ONLY VECTORS CAN BE STORED CURRENTLY. The storage
        required to store entire trained models is large. 

        Parameters
        -----------
        corpus_path (str) - path to the corpus you wish to train the model with
        
        pmodel_name (str) - the name to be assigned to the model when saved. Must be unique
        or the method will return without creating the model to avoid overwriting an existing model

        Returns:
        -----------
        True if model created/trained, False if could not be created

        Throws
        -----------
        FileNotFoundError - If corpus_path not found
        RuntimeError - If training an already existing model that makes it past first if statement. This
        is because build_vocab raises RuntimeError if building existing vocab without update=True (see update_model)
        """
        if installdir != '':
            model_path = installdir + IKWord2VecModeling.__MODEL_PATH_PREFIX__
            vector_path = installdir + IKWord2VecModeling.__VECTOR_PATH_PREFIX__
        
        if pmodel_name[-4:] != '.bin':
                pmodel_name = pmodel_name + '.bin'

        # check if same name model exists by checking for vectors because vectors are always saved
        if os.path.exists(os.path.join(vector_path, pmodel_name)):
            raise FileExistsError("Model named {} already exists, model could not be created".format(pmodel_name[:-4]))
        
        model = Word2Vec(size=psize, sg=1, min_count=pmin_count)

        super().create_new_model(corpus_path, model, epochs)

        if updateable:
            model.save(os.path.join(model_path, pmodel_name[:-4]))

        model.wv.save_word2vec_format(os.path.join(vector_path, pmodel_name), binary=True)
        return True


    @classmethod
    def update_model(cls, corpus_path, pmodel_name, use_iknow_entities=True, tokenize_concepts=True, installdir=''):
        """ Updates an already existing model by continuing its training
        on a new corpus.

        Parameters
        -----------
        corpus_path (str) - path to the corpus being used to update the model
        
        pmodel_name (str, optional) - The name of the model to be updated, defaults to the
        model currently in use

        Return
        -----------
        True if model was updated, else False

        Throws
        -----------
        FileNotFoundError - if corpus or model not found
        """
        if installdir != '':
            model_path = installdir + IKWord2VecModeling.__MODEL_PATH_PREFIX__
            vector_path = installdir + IKWord2VecModeling.__VECTOR_PATH_PREFIX__

        try:
            if pmodel_name[-4:] != '.bin':
                pmodel_name = pmodel_name + '.bin'
            model = Word2Vec.load(os.path.join(model_path, pmodel_name[:-4]))
            
            super().update_model(corpus_path, model, use_iknow_entities, tokenize_concepts)

            # Clear current contents of folders storing model and KeyedVectors files as gensim doesn't do it
            os.remove(os.path.join(model_path, pmodel_name[:-4]))
            os.remove(os.path.join(vector_path, pmodel_name))
            
            model.save(fname_or_handle=os.path.join(model_path, pmodel_name[:-4]))
            model.wv.save_word2vec_format(os.path.join(vector_path, pmodel_name), binary=True)
            
        except FileNotFoundError as err:
            raise FileNotFoundError("Model could not be updated, check specified corpus and model names") from err



class SentenceIterator(object):
    """ An iterator to handle a training corpus that is on multiple files with a given
        directory. 

        This is necessary to interface with gensim when your corpus is split across files.

        Parameters
        -----------
        corpus_path (str) - A path to a file or a directory containing multiple corpus files

        use_iknow_entities (bool) - If True, preprocess lines of the corpus using iKnow enginge so that
        only path relevant entities will be included for training

        tokenize_concepts (bool) - If True, when preprocessing lines from the corpus with iKnow engine, 
        replace concepts with singular tokens (e.g. heart attack -> heart_attack)
    """
    def __init__(self, corpus_path, use_iknow_entities, tokenize_concepts):
        self.corpus_path = corpus_path
        self.is_dir = os.path.isdir(corpus_path)
        if use_iknow_entities:
            self.use_iknow_entities = use_iknow_entities
            self.tokenize_concepts = tokenize_concepts
            self.engine = iknowpy.iKnowEngine()
 
    def __iter__(self):
        if self.is_dir:
            for fname in os.listdir(self.corpus_path):
                for line in open(os.path.join(self.corpus_path, fname)):
                    if self.use_iknow_entities:
                        self.engine.index(line, 'en')
                        for s in self.engine.m_index['sentences']:
                            sent = []
                            for p in s['path']:
                                if s['entities'][p]['type'] == 'Concept' and self.tokenize_concepts:
                                    sent.append(s['entities'][p]['index'].replace(" ", "_")) 
                                else:
                                    sent.append(s['entities'][p]['index'])
                            yield sent
                    else:
                        yield line.split()
        else:
            for line in open(self.corpus_path):
                if self.use_iknow_entities:
                    self.engine.index(line, 'en')
                    sent = []
                    for s in self.engine.m_index['sentences']:
                        for p in s['path']:
                            if s['entities'][p]['type'] == 'Concept' and self.tokenize_concepts:
                                sent.append(s['entities'][p]['index'].replace(" ", "_")) 
                            else:
                                sent.append(s['entities'][p]['index'])
                else:
                    yield line.split()
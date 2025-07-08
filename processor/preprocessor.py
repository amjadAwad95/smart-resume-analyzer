import nltk
import spacy
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class Preprocessor:
    """
    This class provides methods to perform text preprocessing including tokenization,
     stopword removal, lemmatization, and basic text cleaning.
    """
    def __init__(self, nltk_resource="all", spacy_model="en_core_web_sm", language="english"):
        """
        The constructor for download the nltk resource and spacy model.
        :param nltk_resource: nltk resource
        :param spacy_model: spacy model
        :param language: the main language
        """
        try:
            if nltk_resource == "all":
                nltk.data.find(f"corpora")
            else:
                nltk.data.find(f"corpora/{nltk_resource}")

        except LookupError:
            nltk.download(nltk_resource)

        if not spacy.util.is_package(spacy_model):
            spacy.cli.download(spacy_model)

        self.nlp = spacy.load(spacy_model)
        self.stop_word = set(stopwords.words(language))


    def preprocess(self, text):
        """
        This method performs text cleaning, tokenization, and lemmatization.
        :param text: the text to be preprocessed
        :return: the preprocessed text
        """
        text = re.sub(r'(\w+):', r'\1:\n', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \n\2', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()

        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_word]

        doc = self.nlp(" ".join(tokens))
        tokens = [token.lemma_ for token in doc]

        return " ".join(tokens)







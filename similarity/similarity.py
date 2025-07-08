import torch
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizerFast, BertModel
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers.util import cos_sim

class TFIDFSimilarity:
    """
    A class for computing sentence similarity using a TFIDF model.
    """
    def __init__(self):
        """
        Initializes the TfidfVectorizer.
        """
        self.model = TfidfVectorizer(ngram_range=(1, 1))

    def encode(self, sentence1, sentence2):
        """
        Encode the sentence1 and sentence2 using the TFIDF model.
        :param sentence1: The first sentence.
        :param sentence2: The second sentence.
        :return: The encode vector of the sentence1 and sentence2.
        """
        encodes = self.model.fit_transform([sentence1, sentence2])
        return encodes[0], encodes[1]


    def similarity(self, sentence1, sentence2):
        """
        Calculates cosine similarity between two sentence.
        :param sentence1: First sentence text.
        :param sentence2: Second sentence text.
        :return: Cosine similarity score between the two sentence.
        """
        embedding1, embedding2 = self.encode(sentence1, sentence2)

        return cosine_similarity(embedding1, embedding2)[0][0]



class SentenceTransformerSimilarity:
    """
    A class for computing sentence similarity using a SentenceTransformer model.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initializes the SentenceTransformerSimilarity with a given model.
        :param model_name: The name of the pretrained SentenceTransformer model
        """
        self.model = SentenceTransformer(model_name)


    def encode(self, sentence):
        """
        Encodes a sentence into a dense vector representation.
        :param sentence: Input sentence to encode.
        :return: Embedding vector of the sentence.
        """
        return self.model.encode(sentence)


    def similarity(self, sentence1, sentence2):
        """
        Calculates cosine similarity between two sentence.
        :param sentence1: First sentence text.
        :param sentence2: Second sentence text.
        :return: Cosine similarity score between the two sentence.
        """
        embedding1 = self.encode(sentence1)
        embedding2 = self.encode(sentence2)

        return cos_sim(embedding1, embedding2).item()



class BertSimilarity:
    """
    A class for computing sentence similarity using a pretrained BERT model.
    """
    def __init__(self, model_name="google-bert/bert-base-uncased"):
        """
        Initializes the BertSimilarity with a given BERT model.
        :param model_name: The name of the pretrained BERT model.
        """
        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)


    def encode(self, sentence):
        """
        Encodes a sentence into a dense vector using the average of the last 4 hidden layers.
        :param sentence: Input sentence to encode.
        :return: Sentence embedding tensor.
        """
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        last_4_layers = outputs.hidden_states[-4:]

        layer_pooled = [torch.mean(layer, dim=1) for layer in last_4_layers]

        sentence_embedding = torch.mean(torch.stack(layer_pooled), dim=0)

        return sentence_embedding


    def similarity(self, sentence1, sentence2):
        """
        Calculates cosine similarity between two sentence.
        :param sentence1: First sentence text.
        :param sentence2: Second sentence text.
        :return: Cosine similarity score between the two sentence.
        """
        embedding1 = self.encode(sentence1)
        embedding2 = self.encode(sentence2)

        return F.cosine_similarity(embedding1, embedding2).item()
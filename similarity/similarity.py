import torch
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizerFast, BertModel
import torch.nn.functional as F

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


    def similarity(self, embedding1, embedding2):
        """
        Calculates cosine similarity between two embeddings.
        :param embedding1: First sentence embedding.
        :param embedding2: Second sentence embedding.
        :return: Cosine similarity score between the two embeddings.
        """
        return self.model.similarity(embedding1, embedding2).item()


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


    def similarity(self, embedding1, embedding2):
        """
        Calculates cosine similarity between two embeddings.
        :param embedding1: First sentence embedding.
        :param embedding2: Second sentence embedding.
        :return: Cosine similarity score between the two embeddings.
        """
        return F.cosine_similarity(embedding1, embedding2).item()
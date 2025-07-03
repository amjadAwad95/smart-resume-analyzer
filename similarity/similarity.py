import torch
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizerFast, BertModel
import torch.nn.functional as F

class SentenceTransformerSimilarity:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)


    def encode(self, sentence):
        return self.model.encode(sentence)


    def similarity(self, embedding1, embedding2):
        return self.model.similarity(embedding1, embedding2).item()


class BertSimilarity:
    def __init__(self, model_name="google-bert/bert-base-uncased"):
        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)


    def encode(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        last_4_layers = outputs.hidden_states[-4:]

        layer_pooled = [torch.mean(layer, dim=1) for layer in last_4_layers]

        sentence_embedding = torch.mean(torch.stack(layer_pooled), dim=0)

        return sentence_embedding


    def similarity(self, embedding1, embedding2):
        return F.cosine_similarity(embedding1, embedding2).item()
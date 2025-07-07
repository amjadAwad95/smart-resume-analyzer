import spacy
from rapidfuzz import process
from nltk import ngrams
from nltk.tokenize import word_tokenize

class SkillListMatcher:
    """
    Provides methods to extract and match skills from text.
    """
    def __init__(self, spacy_model= "en_core_web_sm"):
        """
        Initializes the matcher and loads the spaCy model.
        :param spacy_model: Name of the spaCy model to load.
        """
        if not spacy.util.is_package(spacy_model):
            spacy.cli.download(spacy_model)

        self.nlp = spacy.load(spacy_model)


    def __lemmatization(self, skills):
        """
        Lemmatizes a list of skills.
        :param skills: List of skill strings.
        :return: List of lemmatized skills.
        """
        new_skills = []
        for i in range(len(skills)):
            skill = skills[i]
            doc = self.nlp(skill)
            tokens = [token.lemma_ for token in doc]
            new_skills.append(" ".join(tokens).lower().strip())

        return new_skills


    def extract(self, text, skills, threshold=95):
        """
        Extracts relevant skills from the given text.
        :param text: The input text.
        :param skills: List of reference skill strings.
        :param threshold: Threshold for matching skills.
        :return: List of matched skills found in the text.
        """
        text = text.lower()
        tokens = word_tokenize(text)

        candidates = set()
        for n in range(1, 5):
            for gram in ngrams(tokens, n):
                phrase = ' '.join(gram)
                candidates.add(phrase)

        new_skills = self.__lemmatization(skills)

        found_skills = set()
        for phrase in candidates:
            match, score, _ = process.extractOne(phrase, new_skills)
            if score >= threshold:
                found_skills.add(match)

        return list(found_skills)


    def match(self, main_skills, extract_skills):
        """
        Matches extracted skills with main skills.
        :param main_skills: List of target skill strings.
        :param extract_skills: List of extracted skill strings.
        :return: Tuple of match ratio and formatted match string.
        """
        main_skills = self.__lemmatization(main_skills)
        extract_skills = self.__lemmatization(extract_skills)

        count = 0

        for skill in extract_skills:
            if skill in main_skills:
                count += 1

        return count / len(main_skills), f"{count}/{len(main_skills)}"


class SkillDynamicMatcher:
    def __init__(self, model_path="skill/Models/model-best"):
        self.ner_model = spacy.load(model_path)


    def extract(self, text):
        skills = []
        doc = self.ner_model(text)

        for ent in doc.ents:
            if "SKILLS" in ent.label_:
                skills.append(ent.text.lower())

        return list(set(skills))


    def match(self, main_skills, extract_skills):
        """
        Matches extracted skills with main skills.
        :param main_skills: List of target skill strings.
        :param extract_skills: List of extracted skill strings.
        :return: Tuple of match ratio and formatted match string.
        """
        count = 0

        for skill in extract_skills:
            if skill in main_skills:
                count += 1

        return count / len(main_skills), f"{count}/{len(main_skills)}"
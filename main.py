from parser import PDFExtractor,TextExtractor
from processor import Preprocessor
from skill import SkillListMatcher
from similarity import SentenceTransformerSimilarity, BertSimilarity

pdf_extractor = PDFExtractor()
text_extractor = TextExtractor()
preprocessor = Preprocessor()
skill_matcher = SkillListMatcher()
sentence_transformer = SentenceTransformerSimilarity()
bert_similarity = BertSimilarity("slone/bert-base-multilingual-cased-bak-rus-similarity")

resume = pdf_extractor.extract("sample_data/amjadCVAI.pdf")
job_description = text_extractor.extract("sample_data/job_description")

resume = preprocessor.preprocess(resume)
job_description = preprocessor.preprocess(job_description)

skills = [
    "Problem solving",
    "Research & Development",
    "Team Collaboration",
    "Leadership",
    "Fast learner",
    "Communication skills",
    "machine learning"
]

match_skills = skill_matcher.extract(resume, skills)

print(resume)
print("----------------------")
print(job_description)
print("----------------------")

print(skill_matcher.match(skills, match_skills))
print("----------------------")

resume_embedding = sentence_transformer.encode(resume)
job_description_embedding = sentence_transformer.encode(job_description)

bert_resume_embedding = bert_similarity.encode(resume)
bert_job_description_embedding = bert_similarity.encode(job_description)

print(sentence_transformer.similarity(resume_embedding, job_description_embedding))
print(bert_similarity.similarity(bert_resume_embedding, bert_job_description_embedding))

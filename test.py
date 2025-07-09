from parser import PDFExtractor,TextExtractor
from processor import Preprocessor
from skill import SkillListMatcher, SkillDynamicMatcher
from similarity import SentenceTransformerSimilarity, BertSimilarity, TFIDFSimilarity
from recommendation import AiRecommendation

pdf_extractor = PDFExtractor()
text_extractor = TextExtractor()
preprocessor = Preprocessor()
skill_matcher = SkillListMatcher()
sentence_transformer = SentenceTransformerSimilarity("mixedbread-ai/mxbai-embed-large-v1")
bert_similarity = BertSimilarity()
tfidf_similarity = TFIDFSimilarity()
# recommendation = AiRecommendation()
matcher = SkillDynamicMatcher()

resume = pdf_extractor.extract("sample_data/amjadCVAI.pdf")
job_description = text_extractor.extract("sample_data/job_description.txt")

resume_skill= matcher.extract(resume)
print("----------------")
jd_skill= matcher.extract(job_description)

resume = preprocessor.preprocess(resume)
job_description = preprocessor.preprocess(job_description)

print(matcher.match(jd_skill, resume_skill))

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

print(sentence_transformer.similarity(job_description, resume))
print(bert_similarity.similarity(job_description,resume ))
print(tfidf_similarity.similarity(job_description, resume))
# print(recommendation.recommend(resume, job_description))
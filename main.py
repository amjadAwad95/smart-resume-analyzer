from parser import PDFExtractor,TextExtractor
from processor import Preprocessor
from skill import SkillListMatcher

pdf_extractor = PDFExtractor()
text_extractor = TextExtractor()
preprocessor = Preprocessor()
skill_matcher = SkillListMatcher()

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
from parser import PDFExtractor,TextExtractor
from processor import Preprocessor

pdf_extractor = PDFExtractor()
text_extractor = TextExtractor()
preprocessor = Preprocessor()

resume = pdf_extractor.extract("sample_data/amjadCVAI.pdf")
job_description = text_extractor.extract("sample_data/job_description")

resume = preprocessor.preprocess(resume)
job_description = preprocessor.preprocess(job_description)

print(resume)
print("----------------------")
print(job_description)
from parser import PDFExtractor,TextExtractor

pdf_extractor = PDFExtractor()
text_extractor = TextExtractor()

resume = pdf_extractor.extract("sample_data/amjadCVAI.pdf")
job_description = text_extractor.extract("sample_data/job_description")

print(resume)
print("----------------------")
print(job_description)
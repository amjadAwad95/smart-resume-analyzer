# Smart Resume Analyzer

Build a system that analyzes resumes using NLP techniques, including preprocessing, skill extraction, and similarity scoring.

## Abstraction

The Smart Resume Analyzer is an AI-driven application that streamlines the resume screening process by leveraging natural language processing (NLP). It enables HR professionals and recruiters to efficiently evaluate candidate resumes by comparing them against specific job descriptions.

## Introduction

The Smart Resume Analyzer is an AI-powered application that simplifies and enhances the resume screening process. By leveraging natural language processing (NLP) techniques, the system extracts key information from resumes, identifies relevant skills, and compares them against job descriptions using semantic similarity. This allows HR professionals and recruiters to make faster, more informed hiring decisions with greater accuracy and consistency.

## Methodology

We followed a systematic process starting with extracting data from resumes and job descriptions, then preprocessing the extracted files, followed by skill extraction from resumes to match job requirements. Finally, we computed similarity scores and provided recommendations for improvement.

### Extract

The extraction stage implements methods to extract text from PDF, DOCX, and plain text files.

### Preprocessor

The preprocessing stage implements various data cleaning techniques using the NLTK and spaCy libraries. We perform text cleaning, convert all characters to lowercase, and tokenize sentences. In the final step, we apply lemmatization to the tokens to reduce words to their base forms.

### Skill Matching

In the skill-matching stage, we extract skills from the resume by first lemmatizing the job skill list. We then generate n-grams ranging from 1 to 5 tokens to better capture skills expressed as single or multiple words. To accurately extract skills, we use an edit distance strategy with a similarity threshold of 95%. Finally, we implement a matching function that compares the two skill lists and calculates the percentage of similarity between them.

### Similarity

In the similarity stage, we implement three different approaches. The first uses a basic vectorization technique with TF-IDF combined with cosine similarity as the metric. The second approach employs a sentence transformer based on the BERT model, also using cosine similarity. The third method utilizes a BERT model where the embeddings are obtained by mean pooling over the last four attention layers, with cosine similarity again serving as the similarity metric.

### Recommendation

In the recommendation stage, we use a pre-trained model from Hugging Face and craft effective prompts to generate meaningful recommendations.

### Evaluation

In the evaluation stage, we test all three methods developed in the similarity stage to determine which performs best. We load a dataset, preprocess and transform the data, and then evaluate the models using standard metrics such as F1-score, accuracy, recall, and precision.

## Results

We evaluated three similarity methods: **TF-IDF**, **Sentence Transformer**, and **BERT with 4-layer mean pooling**. Below are the results based on four evaluation metrics: **accuracy**, **precision**, **recall**, and **F1-score**.

| Metric    | BERT (4-layer pooling) | TF-IDF | Sentence Transformer |
|-----------|------------------------|--------|-----------------------|
| Accuracy  | 0.5128                 | 0.4866 | **0.5549**            |
| Precision | 0.5128                 | 0.0000 | **0.5573**            |
| Recall    | **1.0000**             | 0.0000 | 0.6419                |
| F1-score  | **0.6779**             | 0.0000 | 0.5966                |

- The **BERT model with mean pooling** achieved the highest recall and F1-score, making it the most effective in identifying relevant skills.
- The **Sentence Transformer** model achieved the highest precision and accuracy, indicating more balanced performance.
- The **TF-IDF** method performed poorly across all metrics, highlighting its limitations for semantic comparison in this task.

## Features

- Upload resume and job description (PDF, DOCX, TXT)
- Extract text and preprocess content
- Add and match custom skills
- Visual feedback on skill matching
- Resume-JD similarity score using SentenceTransformer
- AI-generated recommendations to improve your resume

## Tech Stack

- Python
- Streamlit
- NLP: spaCy, Transformers, SentenceTransformer
- AI Model: HuggingFace `SmolLM2-1.7B-Instruct`

## How to Run Locally

1. Clone the repository:
  ```bash
  git clone https://github.com/amjadAwad95/smart-resume-analyzer.git
  cd smart-resume-analyzer
  ```

2. Create a virtual environment:
 ```bash
 python -m venv .venv
 source .venv/bin/activate  # on Windows: .venv\Scripts\activate
 ```
   
3. Install the dependencies:
 ```bash
 pip install -r requirements.txt
 ```
   
4. Run the app:
```bash
streamlit run main.py
```

## Conclusion

The Smart Resume Analyzer demonstrates how modern NLP techniques can significantly enhance the recruitment process by automating resume analysis and skill matching. Through various stages such as preprocessing, skill extraction, and semantic similarity calculation, the system provides a robust way to evaluate candidates based on how well their resumes align with job descriptions.

Among the three similarity methods tested, the BERT model with mean pooling over the last four attention layers achieved the highest F1-score and perfect recall, making it the most effective at identifying relevant skills. While the Sentence Transformer showed the highest accuracy and precision, the TF-IDF method proved to be less reliable for this task.

Overall, the system offers an efficient, scalable, and intelligent solution for resume screening, helping recruiters make more informed and faster hiring decisions.

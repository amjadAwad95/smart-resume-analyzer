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

In the skill-matching stage, we initially used a static method that involved lemmatizing a predefined job skill list and generating n-grams (1 to 5 tokens) from the resume text to capture both single-word and multi-word skills. An edit distance strategy with a 95% similarity threshold was applied to identify close matches, followed by a matching function that calculated the percentage of similarity between the extracted and predefined skills. To improve adaptability and reduce reliance on static lists, we developed a dynamic method by fine-tuning a pre-trained Named Entity Recognition (NER) model using spaCy. This model is capable of extracting skills directly from both resumes and job descriptions based on context, enabling more accurate identification of domain-specific and emerging skills. The dynamic approach offers a more flexible and scalable solution for skill extraction and matching.

### Similarity

In the similarity stage, we implement three different approaches. The first uses a basic vectorization technique with TF-IDF combined with cosine similarity as the metric. The second approach employs a sentence transformer based on the BERT model, also using cosine similarity. The third method utilizes a BERT model where the embeddings are obtained by mean pooling over the last four attention layers, with cosine similarity again serving as the similarity metric.

### Recommendation

In the recommendation stage, we use a pre-trained model from Hugging Face and craft effective prompts to generate meaningful recommendations.

### Evaluation

In the evaluation stage, we test all three methods developed in the similarity stage to determine which performs best. We load a dataset, preprocess and transform the data, and then evaluate the models using standard metrics such as F1-score, accuracy, recall, and precision.

## Results

We evaluated three similarity methods: **TF-IDF**, **Sentence Transformer**, and **BERT with 4-layer mean pooling**. The models were assessed using four evaluation metrics: **accuracy**, **precision**, **recall**, and **F1-score**.

| Metric    | BERT (4-layer pooling) | TF-IDF | Sentence Transformer |
|-----------|------------------------|--------|-----------------------|
| Accuracy  | 0.5128                 | 0.4861 | **0.6032**            |
| Precision | 0.5128                 | 0.0000 | **0.5792**            |
| Recall    | **1.0000**             | 0.0000 | 0.8271                |
| F1-score  | 0.6779                 | 0.0000 | **0.6813**            |

- The **BERT model with mean pooling** achieved perfect recall, making it highly effective at identifying all relevant skills, although with slightly lower precision.
- The **Sentence Transformer** model outperformed all others in terms of **accuracy**, **precision**, and **F1-score**, indicating a more balanced and robust performance.
- The **TF-IDF** method performed poorly across all metrics, reaffirming its limitations for capturing semantic similarity in skill-matching tasks.

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

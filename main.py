import gradio as gr
from parser import PDFExtractor, TextExtractor, DOCXExtractor
from processor import Preprocessor
from skill import SkillDynamicMatcher
from similarity import SentenceTransformerSimilarity
from recommendation import AiRecommendation

# Initialize components
pdf_extractor = PDFExtractor()
docx_extractor = DOCXExtractor()
text_extractor = TextExtractor()
preprocessor = Preprocessor()
skill_matcher = SkillDynamicMatcher()
sentence_transformer = SentenceTransformerSimilarity("mixedbread-ai/mxbai-embed-large-v1")
recommendation = AiRecommendation()


def extract(file):
    if file is None:
        return "No file uploaded."
    file_path = file if isinstance(file, str) else file.name
    if file_path.endswith('.pdf'):
        return pdf_extractor.extract(file_path)
    elif file_path.endswith('.docx'):
        return docx_extractor.extract(file_path)
    elif file_path.endswith('.txt'):
        return text_extractor.extract(file_path)
    else:
        return "Unsupported file type."


def analyze_files(resume_file, job_description_file):
    if not resume_file or not job_description_file:
        return "Please upload both files.", "", "", "", "", ""

    try:
        # Extract and process text
        resume_text = extract(resume_file)
        jd_text = extract(job_description_file)
        preprocess_resume = preprocessor.preprocess(resume_text)
        preprocess_jd = preprocessor.preprocess(jd_text)

        # Skill matching
        matched_jd_skills = skill_matcher.extract(jd_text)
        matched_resume_skills = skill_matcher.extract(resume_text)
        matched_result = skill_matcher.match(matched_jd_skills, matched_resume_skills)

        # Create scrollable skill display
        skill_display = """
        <div style='
            max-height: 300px; 
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            margin-bottom: 15px;
        '>
        """
        for skill in matched_jd_skills:
            if skill in matched_resume_skills:
                skill_display += f"""
                <div style='
                    background-color: #d4edda;
                    color: #155724;
                    padding: 5px 10px;
                    border-radius: 4px;
                    margin: 5px 0;
                    display: inline-block;
                '>✓ {skill}</div>
                """
            else:
                skill_display += f"""
                <div style='
                    background-color: #f8d7da;
                    color: #721c24;
                    padding: 5px 10px;
                    border-radius: 4px;
                    margin: 5px 0;
                    display: inline-block;
                '>✗ {skill}</div>
                """
        skill_display += "</div>"

        # Prepare other outputs
        ratio_text = f"Match Ratio: {matched_result[0]}" if matched_result else "No matches"
        match_string = f"Match Details: {matched_result[1]}" if matched_result else ""
        score = sentence_transformer.similarity(preprocess_resume, preprocess_jd)
        similarity_text = f"Similarity Score: {score:.2f}"

        return resume_text, jd_text, gr.HTML(skill_display), ratio_text, match_string, similarity_text

    except Exception as e:
        return f"Error: {str(e)}", "", "", "", "", ""


def get_ai_recommendation(resume_file, job_description_file):
    if not resume_file or not job_description_file:
        return "Please upload both files first."
    try:
        resume_text = extract(resume_file)
        jd_text = extract(job_description_file)
        return recommendation.recommend(resume_text, jd_text)
    except Exception as e:
        return f"Error: {str(e)}"


# Custom CSS for scrollable containers
custom_css = """
.scrollable-textbox {
    max-height: 300px;
    overflow-y: auto !important;
    border: 1px solid #e0e0e0;
    border-radius: 5px;
    padding: 10px;
}
.scrollable-textbox textarea {
    min-height: 300px !important;
}
"""

with gr.Blocks(title="Resume Analyzer", css=custom_css) as demo:
    gr.Markdown("# 🧠 Smart Resume Analyzer")

    # File upload
    with gr.Row():
        resume_file = gr.File(label="Your Resume", file_types=[".pdf", ".docx", ".txt"])
        job_description_file = gr.File(label="Job Description", file_types=[".pdf", ".docx", ".txt"])

    analyze_btn = gr.Button("Analyze Documents", variant="primary")

    # Results sections
    with gr.Tab("Extracted Text"):
        with gr.Accordion("Resume Content", open=False):
            resume_output = gr.Textbox(
                label="Resume Text",
                lines=20,
                interactive=False,
                elem_classes=["scrollable-textbox"]
            )
        with gr.Accordion("Job Description", open=False):
            jd_output = gr.Textbox(
                label="Job Description Text",
                lines=20,
                interactive=False,
                elem_classes=["scrollable-textbox"]
            )

    with gr.Tab("Analysis Results"):
        gr.Markdown("## Skill Matching")
        skills_output = gr.HTML(label="Skill Comparison")

        with gr.Row():
            ratio_output = gr.Textbox(label="Match Ratio", interactive=False)
            similarity_output = gr.Textbox(label="Similarity Score", interactive=False)

        match_string_output = gr.Textbox(
            label="Detailed Matching",
            interactive=False,
            elem_classes=["scrollable-textbox"]
        )

    with gr.Tab("AI Recommendations"):
        ai_btn = gr.Button("Generate Recommendations", variant="primary")
        ai_output = gr.Textbox(
            label="AI Suggestions",
            lines=20,
            interactive=False,
            elem_classes=["scrollable-textbox"]
        )

    # Event handlers
    analyze_btn.click(
        analyze_files,
        inputs=[resume_file, job_description_file],
        outputs=[resume_output, jd_output, skills_output, ratio_output, match_string_output, similarity_output],
        scroll_to_output=True
    )

    ai_btn.click(
        get_ai_recommendation,
        inputs=[resume_file, job_description_file],
        outputs=[ai_output],
        scroll_to_output=True
    )

demo.launch()
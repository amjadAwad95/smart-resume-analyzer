import streamlit as st
import tempfile
from parser import PDFExtractor, TextExtractor, DOCXExtractor
from processor import Preprocessor
from skill import SkillListMatcher
from similarity import SentenceTransformerSimilarity
from recommendation import AiRecommendation

@st.cache_resource
def get_pdf_extractor():
    return PDFExtractor()

@st.cache_resource
def get_docx_extractor():
    return DOCXExtractor()

@st.cache_resource
def get_text_extractor():
    return TextExtractor()

@st.cache_resource
def get_preprocessor():
    return Preprocessor()

@st.cache_resource
def get_skill_matcher():
    return SkillListMatcher()

@st.cache_resource
def get_sentence_transformer():
    return SentenceTransformerSimilarity()

@st.cache_resource
def get_ai_recommendation():
    return AiRecommendation()

pdf_extractor = get_pdf_extractor()
docx_extractor = get_docx_extractor()
text_extractor = get_text_extractor()
preprocessor = get_preprocessor()
skill_matcher = get_skill_matcher()
sentence_transformer = get_sentence_transformer()
recommendation = get_ai_recommendation()

def extract(file):
    if file.type == "application/pdf":
        return pdf_extractor.extract(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx_extractor.extract(file)
    elif file.type == "text/plain":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        return text_extractor.extract(tmp_path)
    else:
        return "Unsupported file type."

st.set_page_config(page_title="Smart Resume Analyzer", layout="wide")
st.title("🧠 Smart Resume Analyzer")

st.subheader("📁 Upload Files")
col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Resume (PDF/DOCX)", type=["pdf", "docx"])
with col2:
    job_description_file = st.file_uploader("Job Description (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])

if "skills" not in st.session_state:
    st.session_state.skills = []

st.subheader("🛠️ Add Your Skills")
with st.form("skill_form"):
    new_skill = st.text_input("Enter a skill")
    submitted = st.form_submit_button("Add Skill")
    if submitted and new_skill.strip():
        if new_skill.strip() not in st.session_state.skills:
            st.session_state.skills.append(new_skill.strip().lower())


if st.session_state.skills:
    st.success("✅ Your Skills: " + ", ".join(st.session_state.skills))
    if st.button("🧹 Clear All Skills"):
        st.session_state.skills = []
else:
    st.info("No skills added yet.")

if st.button("🔍 Analyze") and resume_file and job_description_file:
    with st.spinner("Processing..."):
        resume_text = extract(resume_file)
        jd_text = extract(job_description_file)

        preprocess_resume = preprocessor.preprocess(resume_text)
        preprocess_jd= preprocessor.preprocess(jd_text)

        matched_skills = skill_matcher.extract(resume_text, st.session_state.skills)
        matched_result = skill_matcher.match(st.session_state.skills, matched_skills)


        st.subheader("📄 Extracted Text")
        with st.expander("Resume"):
            st.write(resume_text)
        with st.expander("Job Description"):
            st.write(jd_text)

        st.subheader("✅ Skill Match")
        cols_per_row = 4
        for i in range(0, len(st.session_state.skills), cols_per_row):
            row = st.columns(cols_per_row)
            for j, skill in enumerate(st.session_state.skills[i:i + cols_per_row]):
                lemma_skills = [token.lemma_ for token in skill_matcher.nlp(skill)]
                lemma_skill = " ".join(lemma_skills)
                with row[j]:
                    if lemma_skill in matched_skills:
                        st.success(f"✓ {skill}")
                    else:
                        st.error(f"✗ {skill}")

        if matched_result:
            st.write(f"ratio {matched_result[0]}")
            st.write(f"formatted match string {matched_result[1]}")
        else:
            st.warning("No matching skills found.")

        st.subheader("📊 Resume-JD Similarity")
        score = sentence_transformer.similarity(preprocess_resume, preprocess_jd)
        st.metric("SentenceTransformer Similarity Score", f"{score:.2f}")

        st.subheader("🤖 AI-Based Recommendation")

        with st.spinner("Generating recommendation..."):
            ai_output = recommendation.recommend(resume_text, jd_text)
            st.info(ai_output)
import os
import time

import streamlit as st
from pypdf import PdfReader
from crewai import Agent, Task, Crew, Process, LLM


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_NAME = "groq/openai/gpt-oss-120b"


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Resume Review Agent",
    page_icon="📄",
    layout="wide",
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_groq_api_key():
    """
    Get the Groq API key from Streamlit secrets.
    """

    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return str(api_key).strip()


def extract_pdf_text(uploaded_file):
    """
    Extract text from an uploaded PDF.
    """

    try:
        reader = PdfReader(uploaded_file)

        if not reader.pages:
            return None, "The PDF does not contain any pages."

        extracted_text = []

        for page in reader.pages:
            try:
                text = page.extract_text()

                if text:
                    extracted_text.append(text)
            except Exception:
                continue

        resume_text = "\n".join(extracted_text).strip()

        if not resume_text:
            return None, (
                "No readable text was found in this PDF. "
                "If it is a scanned/image-only PDF, please paste the resume text."
            )

        return resume_text, None

    except Exception as error:
        return None, f"Could not read the PDF: {error}"


def create_llm(api_key):
    """
    Create the CrewAI LLM using Groq.
    """

    return LLM(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0.2,
        max_tokens=2500,
    )


def run_resume_review(resume_text, job_description, api_key):
    """
    Run the single CrewAI resume-review agent.
    """

    llm = create_llm(api_key)

    reviewer = Agent(
        role="Senior Resume and Job Matching Analyst",

        goal=(
            "Analyze a candidate's resume against a target job description "
            "and provide accurate, evidence-based improvement recommendations "
            "without inventing qualifications, experience, skills, education, "
            "certifications, achievements, or technologies."
        ),

        backstory=(
            "You are an experienced resume reviewer and hiring-support analyst. "
            "You carefully compare resumes with job requirements. "
            "You only use information explicitly present in the candidate's resume. "
            "When something is missing, uncertain, or unsupported, you clearly "
            "say so instead of assuming it is true."
        ),

        llm=llm,

        verbose=False,

        allow_delegation=False,

        max_iter=1,
    )

    task_description = f"""
Review the candidate's resume against the target job description.

IMPORTANT RULES:

1. Never invent information about the candidate.
2. Never assume that the candidate has a skill merely because the job requires it.
3. Only identify a skill, qualification, experience, tool, degree, certification,
   achievement, or technology as present if it is explicitly supported by the resume.
4. Clearly distinguish between:
   - Present evidence
   - Missing requirements
   - Unclear/insufficient evidence
5. Do not rewrite the candidate's experience with fabricated achievements.
6. Do not create fake numbers, metrics, employers, job titles, projects,
   certifications, technologies, or responsibilities.
7. Recommendations must be realistic and actionable.
8. If a requirement is not supported by the resume, recommend adding it only
   if the candidate genuinely has that experience. Otherwise recommend learning
   or gaining the skill rather than falsely claiming it.
9. Do not make a hiring decision.
10. Keep the final response structured and concise.

Return the review using exactly these sections:

# Resume Review

## 1. Overall Match
Give a short explanation of how closely the resume aligns with the job description.
Do not provide a fabricated numerical score.

## 2. Strong Matches
List the important job requirements that are clearly supported by the resume.

For each:
- Requirement
- Resume evidence

## 3. Missing or Weakly Supported Requirements
List important requirements that are:
- clearly missing, or
- not sufficiently supported by the resume.

For each:
- Job requirement
- Evidence status
- Explanation

## 4. Resume Improvement Recommendations
Give practical recommendations for improving the resume.

Focus on:
- wording
- structure
- relevant skills
- projects
- achievements
- keywords
- clarity

Do not invent content.

## 5. Suggested Changes
Give specific examples of how existing resume content could be improved.

Only rewrite information that already exists in the resume.

## 6. Important Warning
Mention any job requirements where the resume does not provide enough evidence.

-------------------------

CANDIDATE RESUME:

{resume_text}

-------------------------

TARGET JOB DESCRIPTION:

{job_description}
"""

    task = Task(
        description=task_description,
        expected_output=(
            "A structured resume review containing match analysis, "
            "supported evidence, missing requirements, actionable "
            "recommendations, and safe suggested changes."
        ),
        agent=reviewer,
    )

    crew = Crew(
        agents=[reviewer],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    return crew.kickoff()


def friendly_error_message(error):
    """
    Convert common API errors into beginner-friendly messages.
    """

    error_text = str(error).lower()

    if (
        "rate limit" in error_text
        or "ratelimit" in error_text
        or "too many requests" in error_text
        or "tokens per day" in error_text
        or "tpm" in error_text
    ):
        return (
            "Groq rate limit reached. Your API key has temporarily reached "
            "its usage limit. Please wait and try again later."
        )

    if (
        "401" in error_text
        or "authentication" in error_text
        or "invalid api key" in error_text
        or "unauthorized" in error_text
    ):
        return (
            "The Groq API key was rejected. Please check your "
            "GROQ_API_KEY in Streamlit Secrets."
        )

    if "timeout" in error_text:
        return (
            "The Groq request timed out. Please try again with a shorter "
            "resume or job description."
        )

    if "context" in error_text or "maximum" in error_text:
        return (
            "The input is too large for the current request. "
            "Please shorten the resume or job description and try again."
        )

    return (
        "The AI review could not be completed. "
        "Please try again. If the problem continues, check the Streamlit logs."
    )


# =========================================================
# USER INTERFACE
# =========================================================

st.title("📄 AI Resume Review Agent")

st.write(
    "Upload or paste your resume, provide a target job description, "
    "and let the AI compare them and suggest improvements."
)

st.info(
    "The agent is instructed not to invent qualifications, experience, "
    "skills, achievements, or certifications."
)


# =========================================================
# API KEY CHECK
# =========================================================

api_key = get_groq_api_key()

if not api_key:
    st.error(
        "Groq API key is missing. Add GROQ_API_KEY to Streamlit Secrets."
    )
    st.stop()


# =========================================================
# RESUME INPUT
# =========================================================

st.subheader("1. Your Resume")

input_method = st.radio(
    "Choose resume input method:",
    ["Paste resume text", "Upload PDF"],
    horizontal=True,
)


resume_text = ""


if input_method == "Paste resume text":

    resume_text = st.text_area(
        "Paste your resume here",
        height=350,
        placeholder=(
            "Paste your complete resume text here..."
        ),
    )

else:

    uploaded_file = st.file_uploader(
        "Upload your resume PDF",
        type=["pdf"],
        help="Upload a text-based PDF resume.",
    )

    if uploaded_file:

        with st.spinner("Reading PDF..."):
            extracted_text, pdf_error = extract_pdf_text(uploaded_file)

        if pdf_error:
            st.error(pdf_error)
        else:
            resume_text = extracted_text

            st.success("PDF text extracted successfully.")

            with st.expander("Preview extracted resume"):
                st.text(resume_text[:5000])


# =========================================================
# JOB DESCRIPTION
# =========================================================

st.subheader("2. Target Job Description")

job_description = st.text_area(
    "Paste the job description here",
    height=300,
    placeholder=(
        "Paste the complete job description here..."
    ),
)


# =========================================================
# REVIEW BUTTON
# =========================================================

st.subheader("3. Review")

review_button = st.button(
    "🔍 Review My Resume",
    type="primary",
    use_container_width=True,
)


if review_button:

    # -----------------------------
    # Validate inputs
    # -----------------------------

    if not resume_text.strip():
        st.warning(
            "Please provide your resume by pasting the text or uploading a PDF."
        )
        st.stop()

    if not job_description.strip():
        st.warning(
            "Please paste the target job description."
        )
        st.stop()

    # -----------------------------
    # Basic size protection
    # -----------------------------

    max_input_chars = 45000

    if len(resume_text) > max_input_chars:
        st.warning(
            "Your resume text is unusually large. "
            "Only the first 45,000 characters will be analyzed."
        )

        resume_text = resume_text[:max_input_chars]

    if len(job_description) > max_input_chars:
        st.warning(
            "The job description is unusually large. "
            "Only the first 45,000 characters will be analyzed."
        )

        job_description = job_description[:max_input_chars]

    # -----------------------------
    # Run agent
    # -----------------------------

    with st.spinner(
        "AI agent is analyzing your resume against the job description..."
    ):

        try:

            result = run_resume_review(
                resume_text=resume_text,
                job_description=job_description,
                api_key=api_key,
            )

            st.success("Resume review completed.")

            st.markdown("## Results")

            st.markdown(str(result))

        except Exception as error:

            st.error(
                friendly_error_message(error)
            )

            with st.expander("Technical details"):
                st.code(str(error))


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Powered by CrewAI + Groq GPT-OSS 120B"
)

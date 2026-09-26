# 📄 AI Resume Review Agent

A beginner-friendly single-agent Resume Review application built with:

- Streamlit
- CrewAI
- Groq
- GPT-OSS 120B
- Python 3.11
- PyPDF

The application compares a candidate's resume with a target job description and provides structured, actionable improvement recommendations.

## Features

- Paste resume text
- Upload a PDF resume
- Paste a target job description
- Compare resume against job requirements
- Identify strong matches
- Identify missing or weakly supported requirements
- Provide resume improvement recommendations
- Suggest improvements to existing resume content
- Avoid inventing qualifications or experience
- Handle PDF extraction errors
- Handle missing inputs
- Handle Groq API errors and rate limits
- Deployable on Streamlit Community Cloud

## Project Structure

```text
resume-review-agent/
│
├── app.py
├── requirements.txt
└── README.md

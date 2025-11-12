import streamlit as st
import os
import PyPDF2
import openai
from agent_graph import run_interview_generator  # your multi-agent pipeline (imported from your project)
import logging
import requests
import asyncio

openai.api_key = "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"
jd_text_for_payload = ""
# -------------------------------
# LOGGING CONFIG
# -------------------------------
# This sets up a simple logger that prints timestamped messages to help you
# understand what the app is doing while it runs.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

api_url = "http://127.0.0.1:8000/users/"  # Replace with your API endpoint
response = requests.get(api_url)

print("---------->")
print(response.text)

# -------------------------------
# STREAMLIT PAGE CONFIG
# -------------------------------
# Sets browser tab title, favicon and page layout.
st.set_page_config(
    page_title="Interview Question Generator",
    page_icon="🤖",
    layout="centered"
)

# -------------------------------
# OPENAI API KEY SETUP
# -------------------------------
# NOTE: This is shown here for demonstration. In real projects DO NOT hardcode keys.
# TIP: Use environment variables or Streamlit secrets (st.secrets["OPENAI_API_KEY"]).
OPENAI_API_KEY = "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"  # <-- placeholder
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

# -------------------------------
# STREAMLIT SESSION STATE
# -------------------------------
# We remember which "mode" the user chose across interactions:
# - "agentic": uses your multi-agent flow (run_interview_generator)
# - "non_agentic": uses a simple, direct prompting approach in this file
if "mode" not in st.session_state:
    st.session_state.mode = "agentic"  # default view

# -------------------------------
# CUSTOM CSS (for a glassy, dark UI)
# -------------------------------
# Streamlit allows injecting CSS to style widgets and containers.
# Everything here is purely presentational.
st.markdown(
    """
    <style>
    .main {
        background: radial-gradient(circle at 20% 20%, #1f2937 0%, #0f172a 60%);
        color: #f8fafc !important;
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
    }
    header[data-testid="stHeader"] { background: rgba(0,0,0,0); }
    footer {visibility: hidden;}
    div.block-container{padding-top:2rem; max-width: 760px;}

    .app-card {
        background: rgba(15,23,42,0.6);
        border: 1px solid rgba(148,163,184,0.2);
        box-shadow: 0 30px 80px rgba(0,0,0,0.6);
        border-radius: 1rem;
        padding: 1.5rem 2rem 2rem;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        color: #f8fafc;
    }

    .app-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: baseline;
        gap: .5rem;
    }

    .app-sub {
        font-size: .8rem;
        color: #94a3b8;
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: .9rem;
        font-weight: 600;
        color: #cbd5e1;
        margin-top: 1rem;
        margin-bottom: .4rem;
    }

    label[data-testid="stWidgetLabel"] > div > p {
        color: #e2e8f0 !important;
        font-weight: 500;
        font-size: .8rem;
    }

    .stAlert div {
        color: #0f172a !important;
        font-weight: 500;
        font-size: .8rem;
    }

    .output-block {
        background: rgba(30,41,59,0.5);
        border: 1px solid rgba(148,163,184,0.2);
        border-radius: .75rem;
        padding: 1rem 1rem .8rem;
        margin-top: 1rem;
    }
    .output-block h3 {
        margin: 0 0 .5rem;
        font-size: .8rem;
        font-weight: 600;
        color: #93c5fd;
        text-transform: uppercase;
        letter-spacing: .05em;
    }
    .output-text {
        font-size: .85rem;
        color: #f1f5f9;
        white-space: pre-wrap;
        word-break: break-word;
        line-height: 1.4rem;
    }

    textarea, input, select {
        background-color: rgba(15,23,42,0.6) !important;
        color: #f8fafc !important;
    }

    /* Primary vs Secondary buttons (used for mode toggle) */
    div[data-testid="stButton"] > button[kind="primary"] {
        width: 100%;
        background: radial-gradient(circle at 0% 0%, #4f46e5 0%, #0ea5e9 80%) !important;
        border: 1px solid rgba(226,232,240,0.4) !important;
        box-shadow: 0 25px 60px rgba(14,165,233,.5) !important;
        color: #fff !important;
        border-radius: .75rem !important;
        padding: .8rem .9rem !important;
        font-size: .8rem !important;
        font-weight: 600 !important;
        line-height: 1.2rem !important;
        text-align: left !important;
    }
    div[data-testid="stButton"] > button[kind="secondary"] {
        width: 100%;
        background: rgba(15,23,42,0.6) !important;
        border: 1px solid rgba(148,163,184,0.3) !important;
        box-shadow: 0 20px 40px rgba(0,0,0,0.6) !important;
        color: #e2e8f0 !important;
        border-radius: .75rem !important;
        padding: .8rem .9rem !important;
        font-size: .8rem !important;
        font-weight: 600 !important;
        line-height: 1.2rem !important;
        text-align: left !important;
    }

    /* Allow multi-line labels inside those buttons */
    div[data-testid="stButton"] > button[kind="primary"] p,
    div[data-testid="stButton"] > button[kind="secondary"] p {
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------
# HELPER: PDF → plain text
# -------------------------------
# Takes a PDF file-like object (from Streamlit uploader) and extracts plain text.
# PyPDF2's extract_text() works best on "real text" PDFs (not scanned images).
def pdf_to_text(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    num_pages = len(pdf_reader.pages)
    text = ""
    for page_num in range(num_pages):
        page_obj = pdf_reader.pages[page_num]
        page_text = page_obj.extract_text() or ""  # fall back to empty if None
        # Basic cleanup: join broken lines and excess spaces
        page_text = (
            page_text.strip()
            .replace("\n", " ")
            .replace("  ", " ")
        )
        text += page_text + " "
    return text.strip()


# -------------------------------
# NON-AGENTIC HELPERS (simple prompts)
# -------------------------------
# These functions call OpenAI's "completions" endpoint to:
# 1) Extract skills from JD and Resume
# 2) Find common skills
# 3) Generate interview questions from those common skills

def extract_skills(text, source):
    """
    Ask the model to list technical skills found in the given text.
    'source' is just a label for logging (e.g., 'Resume' or 'Job Description').
    """
    prompt = f"""Extract key technical skills mentioned in the following {source}. The skills should be related to programming, AI, ML, data science, and other relevant fields.
{text}"""

    logger.info(f"[Non-Agentic] Extracting skills from {source} (prompt {len(prompt)} chars)")
    # NOTE: This uses the legacy Instruct/Completions API (as per your code).
    response = openai.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=0.3,  # low = more deterministic
        max_tokens=500,
    )
    logger.info(f"[Non-Agentic] {source} → received {len(response.choices[0].text)} chars")
    return response.choices[0].text.strip()


def find_common_skills(jd_skills, resume_skills):
    """
    Ask the model to intersect the two skill lists and describe common items.
    """
    prompt = f"""Find the common technical skills between the following JD skills and Resume skills:
                    JD Skills: {jd_skills}
                    Resume Skills: {resume_skills}
            """

    logger.info("[Non-Agentic] Finding common skills ...")
    response = openai.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=0.3,
        max_tokens=200,
    )
    return response.choices[0].text.strip()


def generate_questions(common_skills, num_questions, difficulty_level):
    """
    Ask the model to write N interview questions, tuned by difficulty.
    The prompt forces numbering to start at 1 and be sequential.
    """
    prompt = f"""Based on the following common technical skills, generate {num_questions} interview questions with a difficulty level of {difficulty_level}.
Common Skills: {common_skills}.
The generated question numbers should start from 1. and must be strictly sequential (1., 2., 3., ...). Do not skip numbers. Do not start from 0."""

    logger.info("[Non-Agentic] Generating interview questions ...")
    response = openai.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=0.7,  # a bit more creative for question writing
        max_tokens=1000,
    )
    asyncio.run(post_result_to_db(response.choices[0].text.strip().splitlines()))
    return response.choices[0].text.strip()


async def post_result_to_db(lines):
    #url = "http://127.0.0.1:8000/users/"
    #payload = {"name": "Test", "qa": [], "jd": jd_text_for_payload}  # Example URL for testing POST requests
    for line in lines:
        answer = openai.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=line,
            temperature=0.7,
            max_tokens=50,
        ).choices[0].text.strip()

        print(line)
        print(answer)

        #payload["qa"].append({"question": line, "answer": answer})

    #response = requests.post(url, json=payload)
    #print(response.status_code)
    #print(response.text)


def run_non_agentic_flow(resume_file_like, jd_text, num_q, difficulty):
    """
    Orchestrates the simple, single-model flow:
      1) Convert resume PDF → text
      2) Extract skills from JD and Resume
      3) Compute common skills
      4) Generate interview questions from the common skills
    Returns a dict that the UI will render.
    """
    logger.info(f"[Non-Agentic] Starting flow with difficulty={difficulty}, num_q={num_q}")

    # 1) Read resume PDF and extract raw text
    resume_text = pdf_to_text(resume_file_like)
    logger.info(f"[Non-Agentic] Resume text extracted ({len(resume_text)} chars)")

    # 2) Extract skills separately from JD and Resume
    jd_skills = extract_skills(jd_text, "Job Description")
    resume_skills = extract_skills(resume_text, "Resume")

    logger.info(f"[Non-Agentic] JD skills sample: {jd_skills[:100]}...")
    logger.info(f"[Non-Agentic] Resume skills sample: {resume_skills[:100]}...")

    # 3) Find commonality between both lists
    common_skills = find_common_skills(jd_skills, resume_skills)
    logger.info(f"[Non-Agentic] Common skills: {common_skills}")

    # 4) Generate N questions at chosen difficulty
    questions = generate_questions(common_skills, num_q, difficulty)
    logger.info(f"[Non-Agentic] Generated {len(questions.splitlines())} lines of questions")

    # Return a structured payload for the UI
    return {"common_skills": common_skills, "questions": questions, "skill_report": ""}


# -------------------------------
# APP HEADER
# -------------------------------
# A small header explaining what the app does.
st.markdown(
    """
    <div class="app-card">
        <div class="app-header">
            <span>AI Interview Question Generator</span>
        </div>
        <div class="app-sub">
            Upload a resume + paste a JD → get tailored interview questions.
            Choose between a direct prompt flow or a multi-agent reasoning flow.
        </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# MODE TOGGLE (Non-Agentic vs Agentic)
# -------------------------------
# Two buttons act like radio buttons: whichever is clicked becomes active ("primary").
left_col, right_col = st.columns(2)

with left_col:
    non_agentic_clicked = st.button(
        "🛡️ Non-Agentic\n"
        "direct prompt",
        key="non_agentic_btn",
        type="primary" if st.session_state.mode == "non_agentic" else "secondary",
        help="Direct single-shot prompting (cheap, fast)",
    )

with right_col:
    agentic_clicked = st.button(
        "🧠 Agentic\n"
        "Multi-agent reasoning",
        key="agentic_btn",
        type="primary" if st.session_state.mode == "agentic" else "secondary",
        help="Multi-step chain of agents (richer output)",
    )

# Update session state based on which button the user clicked.
if non_agentic_clicked:
    st.session_state.mode = "non_agentic"
    logger.info("[UI] Mode switched to NON-AGENTIC")
elif agentic_clicked:
    st.session_state.mode = "agentic"
    logger.info("[UI] Mode switched to AGENTIC")

# Close the card opened earlier
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# INPUTS: Resume + JD + Settings
# -------------------------------
# Resume PDF, JD text, number of questions, and difficulty selector.
st.markdown('<div class="section-title">Candidate Inputs</div>', unsafe_allow_html=True)

# File uploader returns a file-like object (buffer) for PDFs
resume_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

# Multiline text area to paste the job description
jd_text = st.text_area(
    "Paste the Job Description",
    height=180,
    placeholder="Paste full JD here..."
)

# Two small columns: one for number of questions, one for difficulty
col1, col2 = st.columns(2)
with col1:
    num_q = st.number_input(
        "Number of Questions",
        min_value=1,
        max_value=30,
        value=10,
        help="How many questions should we create?"
    )
with col2:
    difficulty = st.selectbox(
        "Difficulty",
        ["Easy", "Medium", "Hard"],
        index=1,
        help="Overall difficulty level of the generated questions"
    )

# Main action button to start generation
generate_clicked = st.button(
    "Generate Questions",
    type="primary")

# Close section wrapper
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# MAIN ACTION: Run selected flow
# -------------------------------
# When the user clicks "Generate Questions", we:
# - Validate inputs
# - Run either the simple flow or the multi-agent flow
# - Render outputs (common skills, questions, and optional coach notes)
if generate_clicked:
    # Basic validation: both a PDF and JD text are required
    if resume_file is None or not jd_text.strip():
        st.warning("Please upload a resume and paste the JD.")
    else:
        logger.info(f"[RUN] Button clicked with mode={st.session_state.mode}")

        # Read the raw bytes for the agentic flow (your graph expects bytes)
        resume_bytes = resume_file.read()

        # Decide which flow to run based on selected mode
        if st.session_state.mode == "non_agentic":
            jd_text_for_payload = jd_text
            # Run simple single-prompt flow (all calls are in this file)
            result = run_non_agentic_flow(
                resume_file_like=resume_file,  # file-like object for pdf_to_text()
                jd_text=jd_text,
                num_q=num_q,
                difficulty=difficulty,
            )
        else:
            # Run your external multi-agent pipeline (imported at the top)
            # This likely does more steps (OCR, extraction, reasoning, etc.)
            logger.info("[Agentic] Running multi-agent interview generator ...")
            result = run_interview_generator(
                resume_bytes=resume_bytes,
                jd_text=jd_text,
                num_q=num_q,
                difficulty=difficulty,
            )
            logger.info("[Agentic] Received result from multi-agent graph.")

        # ---------------------------
        # RENDER RESULTS
        # ---------------------------
        # Normalize "common_skills" in case your agent returns a list or a string
        raw_common = result.get("common_skills", [])
        if isinstance(raw_common, list):
            common_display = ", ".join(raw_common) or "None detected"
        else:
            common_display = raw_common or "None detected"

        # The generated questions come back as a single string block (numbered)
        questions_display = result.get("questions", "No questions generated.")
        # Optional guidance from the agent about how to interview this candidate
        skill_report_display = result.get("skill_report", "")

        # Pretty output blocks
        st.markdown(
            '<div class="output-block"><h3>Common Skills</h3>'
            f'<div class="output-text">{common_display}</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="output-block"><h3>Interview Questions</h3>'
            f'<div class="output-text">{questions_display}</div></div>',
            unsafe_allow_html=True,
        )

        if skill_report_display:
            st.markdown(
                '<div class="output-block"><h3>How to Interview This Candidate</h3>'
                f'<div class="output-text">{skill_report_display}</div></div>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    generate_questions("java, spring, springboot", "5", "medium")
import os
import io
import json
import PyPDF2
from typing import TypedDict, Dict, Any, List

from langgraph.graph import StateGraph, END

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class InterviewState(TypedDict, total=False):
    resume_file_bytes: bytes
    jd_text: str
    num_questions: int
    difficulty: str

    resume_text: str
    resume_skills: List[str]
    jd_skills: List[str]
    common_skills: List[str]

    questions: str
    skill_report: str



def extract_pdf_text(pdf_bytes: bytes) -> str:
    """
    Extract text from a PDF (no OCR).
    """
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        txt = " ".join(txt.split())
        pages.append(txt)
    return "\n".join(pages)

def llm_call(
    system_prompt: str,
    user_prompt: str,
    model="gpt-4o-mini",
    temperature=0.3,
    max_tokens=800,
):
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",  "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


def resume_parser_node(state: InterviewState) -> InterviewState:
    """
    Input: resume_file_bytes
    Output: resume_text
    """
    resume_bytes = state["resume_file_bytes"]
    resume_text = extract_pdf_text(resume_bytes)
    return {
        **state,
        "resume_text": resume_text,
    }


def skill_extractor_node(state: InterviewState) -> InterviewState:
    """
    Input: resume_text, jd_text
    Output: resume_skills, jd_skills, common_skills
    """
    jd_text = state["jd_text"]
    resume_text = state["resume_text"]

    system_prompt = (
        "You are a strict skill extraction agent. "
        "You ONLY return valid technical skills, tools, libraries, frameworks, "
        "cloud platforms, programming languages, ML techniques, math/DS concepts, etc. "
        "Return clean JSON. Do not add commentary."
    )

    user_prompt = f"""
Extract skills from BOTH documents.

Return JSON of the form:
{{
  "resume_skills": ["skill1", "skill2", ...],
  "jd_skills": ["skill1", "skill2", ...],
  "common_skills": ["skill_common1", ...]
}}

Rules:
- lowercase everything.
- no soft skills (teamwork, communication).
- combine aliases (aws / amazon web services -> aws).
- only include in common_skills if it appears in BOTH.

--- JOB DESCRIPTION ---
{jd_text}

--- RESUME TEXT ---
{resume_text}
"""

    raw = llm_call(system_prompt, user_prompt, temperature=0.2, max_tokens=700)

    try:
        parsed = json.loads(raw)
    except:
        fix_prompt = f"""Fix and return ONLY valid JSON for the following attempt:\n{raw}"""
        fixed = llm_call(
            "You repair malformed JSON. Output ONLY valid minified JSON. No commentary.",
            fix_prompt,
            temperature=0.0,
            max_tokens=400,
        )
        parsed = json.loads(fixed)

    resume_skills = parsed.get("resume_skills", [])
    jd_skills = parsed.get("jd_skills", [])
    common_skills = parsed.get("common_skills", [])

    return {
        **state,
        "resume_skills": resume_skills,
        "jd_skills": jd_skills,
        "common_skills": common_skills,
    }


def gap_analysis_node(state: InterviewState) -> InterviewState:
    """
    Input: resume_skills, jd_skills, common_skills
    Output: skill_report (interviewer guidance)
    """
    resume_skills = state.get("resume_skills", [])
    jd_skills = state.get("jd_skills", [])
    common_skills = state.get("common_skills", [])

    system_prompt = (
        "You are an interview planning analyst. "
        "You advise a hiring manager how to evaluate this candidate."
    )

    user_prompt = f"""
You are given:
- JD skills: {jd_skills}
- Candidate resume skills: {resume_skills}
- Overlap (common skills): {common_skills}

Produce 3 sections in plain text, short bullets, not JSON:

1. CORE MATCH
   - Skills where candidate matches JD.
   - For each, say what kind of deep technical probing to do.

2. POTENTIAL GAPS
   - Skills JD wants but resume doesn't clearly show.
   - For each gap, give 1 friendly screening question
     to test if the candidate actually has that skill.

3. RISK FLAGS
   - Any areas that feel buzzword-y or shallow
     and should be double-checked.
   - Keep this respectful and factual.
"""

    report = llm_call(system_prompt, user_prompt, temperature=0.4, max_tokens=600)

    return {
        **state,
        "skill_report": report,
    }


def question_gen_node(state: InterviewState) -> InterviewState:
    """
    Draft technical questions based on common skills.
    Input: common_skills, num_questions, difficulty
    Output: questions (raw draft)
    """
    common_skills = state.get("common_skills", [])
    num_q = state["num_questions"]
    difficulty = state["difficulty"]

    system_prompt = (
        "You are an interview question generator. "
        "You produce concise, technically relevant questions."
    )

    user_prompt = f"""
Generate {num_q} interview questions.
Difficulty: {difficulty}.
Focus ONLY on these skills: {common_skills}.

Rules:
- Number questions starting from 1.
- No sub-bullets, just a flat list.
- Avoid yes/no questions.
- Prefer scenario / debugging / design style questions.
- Keep each question answerable verbally in 2-3 minutes.
"""

    questions = llm_call(system_prompt, user_prompt, temperature=0.7, max_tokens=1200)

    return {
        **state,
        "questions": questions,
    }


def question_reviewer_node(state: InterviewState) -> InterviewState:
    """
    Improve clarity/consistency of questions.
    Input: questions, difficulty
    Output: refined questions
    """
    raw_questions = state.get("questions", "")
    difficulty = state.get("difficulty", "")

    system_prompt = (
        "You are a senior technical interviewer. "
        "You review the draft interview questions and improve them."
    )

    user_prompt = f"""
Here are the draft questions for difficulty level "{difficulty}":

{raw_questions}

Task:
1. Remove duplicates.
2. Keep the SAME total number of questions.
3. All questions must be scenario-based, debugging-based,
   or design-based (not trivia).
4. Keep numbering starting from 1.
5. Make sure each question maps to at least one of the candidate's common skills.

Return ONLY the improved numbered list.
"""

    improved = llm_call(system_prompt, user_prompt, temperature=0.5, max_tokens=1000)

    return {
        **state,
        "questions": improved,
    }



def build_graph():
    workflow = StateGraph(InterviewState)

    workflow.add_node("resume_parser",      resume_parser_node)
    workflow.add_node("skill_extractor",    skill_extractor_node)
    workflow.add_node("gap_analysis",       gap_analysis_node)
    workflow.add_node("question_gen",       question_gen_node)
    workflow.add_node("question_reviewer",  question_reviewer_node)

    workflow.set_entry_point("resume_parser")

    workflow.add_edge("resume_parser",     "skill_extractor")
    workflow.add_edge("skill_extractor",   "gap_analysis")
    workflow.add_edge("gap_analysis",      "question_gen")
    workflow.add_edge("question_gen",      "question_reviewer")
    workflow.add_edge("question_reviewer", END)

    app = workflow.compile()
    return app


graph_app = build_graph()


def run_interview_generator(
    resume_bytes: bytes,
    jd_text: str,
    num_q: int,
    difficulty: str
) -> Dict[str, Any]:
    """
    Kicks off the LangGraph app with initial state and returns final state.
    """
    initial_state: InterviewState = {
        "resume_file_bytes": resume_bytes,
        "jd_text": jd_text,
        "num_questions": num_q,
        "difficulty": difficulty,
    }

    final_state = graph_app.invoke(initial_state)

    return {
        "resume_skills":  final_state.get("resume_skills", []),
        "jd_skills":      final_state.get("jd_skills", []),
        "common_skills":  final_state.get("common_skills", []),
        "questions":      final_state.get("questions", ""),
        "skill_report":   final_state.get("skill_report", ""),
    }

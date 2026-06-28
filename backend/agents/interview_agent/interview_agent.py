"""
Interview Agent Module for CareerPilot-AI.

This module contains the InterviewAgent class along with Pydantic schemas for
interview sessions and feedback. The InterviewAgent is responsible for generating
tailored mock interview questions, grading candidate answers, scoring responses,
and producing comprehensive feedback reports.

Compatible with Google ADK framework, designed to support WebSocket turn-based loops
and Model Context Protocol (MCP) registrations.
"""

import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# Graceful import check for Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# Graceful import check for Google ADK
try:
    # pyrefly: ignore [missing-import]
    from google.adk.agents import Agent as AdkAgent
    HAS_ADK = True
except ImportError:
    HAS_ADK = False


# =====================================================================
# 1. Pydantic Schemas
# =====================================================================

class InterviewQuestion(BaseModel):
    """Schema representing an interview question generated for the candidate."""
    
    question_id: int = Field(..., description="Unique integer identifier for the question")
    question_text: str = Field(..., description="The actual question string asked to the candidate")
    question_type: str = Field(
        ..., 
        description="Category of the question: 'technical', 'behavioral', or 'system_design'"
    )
    target_skill: str = Field(..., description="The specific skill or attribute being assessed")
    expected_points: List[str] = Field(
        default_factory=list,
        description="Key points or keywords that a high-quality answer should cover"
    )
    difficulty: str = Field("medium", description="Question difficulty: 'easy', 'medium', 'hard'")


class InterviewFeedback(BaseModel):
    """Schema representing structured feedback for a single answer."""
    
    question_id: int = Field(..., description="Identifier matching the corresponding question")
    score: int = Field(
        ..., 
        description="Score graded on a scale of 0 to 10",
        ge=0,
        le=10
    )
    strengths: List[str] = Field(
        default_factory=list, 
        description="Good points, technical accuracy, or context identified in the user's answer"
    )
    weaknesses: List[str] = Field(
        default_factory=list, 
        description="Gaps, incorrect claims, or missed key concepts in the user's answer"
    )
    suggested_improvements: List[str] = Field(
        default_factory=list, 
        description="Actionable advice on how to phrase or structure the answer better next time"
    )
    model_answer: str = Field(
        ..., 
        description="An ideal, expert-level response exemplifying the required knowledge"
    )


class InterviewSession(BaseModel):
    """Schema representing the status and accumulated logs of an active mock interview."""
    
    session_id: str = Field(..., description="Unique UUID or token for the interview session")
    candidate_name: str = Field(..., description="Name of the candidate")
    target_role: str = Field(..., description="Target job title or role description")
    questions: List[InterviewQuestion] = Field(default_factory=list)
    answers: Dict[int, str] = Field(
        default_factory=dict, 
        description="History of submitted answers mapped by question_id"
    )
    feedback_by_question: Dict[int, InterviewFeedback] = Field(
        default_factory=dict, 
        description="Grades and feedback summaries mapped by question_id"
    )
    overall_score: Optional[float] = Field(
        None, 
        description="Weighted average score of the session out of 100",
        ge=0.0,
        le=100.0
    )
    overall_feedback: Optional[str] = Field(
        None, 
        description="High-level evaluation summarizing communication, accuracy, and preparation advice"
    )
    is_completed: bool = Field(False, description="Flag indicating whether all questions have been graded")


# =====================================================================
# 2. InterviewAgent Class
# =====================================================================

class InterviewAgent:
    """
    Interview Agent specialized in conducting interactive mock interviews,
    generating relevant questions, evaluating answers turn-by-turn, and scoring sessions.
    
    Designed to hook into WebSocket loops and function under Google ADK.
    """

    def __init__(self, model_name: str = "gemini-2.5-pro", api_key: Optional[str] = None):
        """
        Initializes the InterviewAgent.

        Args:
            model_name: The Gemini model name to use. Defaults to gemini-2.5-pro (highly recommended for grading).
            api_key: Optional Gemini API key. Defaults to environment variable.
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._client = None

        if HAS_GENAI and self.api_key:
            self._client = genai.Client(api_key=self.api_key)

    def _generate_fallback_questions(
        self, 
        resume_skills: List[str], 
        target_role: str, 
        limit: int
    ) -> List[InterviewQuestion]:
        """Provides a standard set of baseline questions based on candidate skills."""
        fallback_bank = [
            InterviewQuestion(
                question_id=1,
                question_text=f"Can you tell me about a time you solved a complex challenge in your role as a {target_role}?",
                question_type="behavioral",
                target_skill="Problem Solving",
                expected_points=["STAR method", "clear context", "technical metrics", "resolution"],
                difficulty="medium"
            ),
            InterviewQuestion(
                question_id=2,
                question_text="How do you handle managing code quality and peer review structures in a collaborative team?",
                question_type="behavioral",
                target_skill="Collaboration",
                expected_points=["constructive feedback", "git workflows", "linting", "testing standards"],
                difficulty="easy"
            ),
            InterviewQuestion(
                question_id=3,
                question_text="Explain the architectural trade-offs between implementing RESTful APIs vs. using WebSockets or gRPC.",
                question_type="technical",
                target_skill="System Design",
                expected_points=["statelessness", "network overhead", "realtime bidirection", "connection persistence"],
                difficulty="hard"
            )
        ]

        # Attempt to tailor a question based on extracted skills
        skills_matched = [s for s in resume_skills if s.lower() in ["python", "react", "sql", "aws", "docker"]]
        if skills_matched:
            skill = skills_matched[0]
            fallback_bank.append(
                InterviewQuestion(
                    question_id=4,
                    question_text=f"Describe memory management, caching patterns, or concurrency in {skill}.",
                    question_type="technical",
                    target_skill=skill,
                    expected_points=["garbage collection", "performance threads", "locks/state", "caching layers"],
                    difficulty="hard"
                )
            )

        # Truncate or pad to match the requested limit
        result = fallback_bank[:limit]
        while len(result) < limit:
            new_id = len(result) + 1
            result.append(
                InterviewQuestion(
                    question_id=new_id,
                    question_text=f"What is your approach to handling scalability and deployment for a modern {target_role} application?",
                    question_type="technical",
                    target_skill="DevOps",
                    expected_points=["horizontal scaling", "load balancers", "containers", "ci/cd pipelines"],
                    difficulty="medium"
                )
            )
        return result

    def _evaluate_answer_locally(
        self, 
        question: InterviewQuestion, 
        user_answer: str
    ) -> InterviewFeedback:
        """Evaluates user answer locally based on expected points matching (fallback)."""
        answer_lower = user_answer.lower()
        matched_points = []
        missed_points = []

        for pt in question.expected_points:
            pt_words = pt.lower().split()
            # If any significant word matches, mark as mentioned
            if any(w in answer_lower for w in pt_words if len(w) > 3):
                matched_points.append(pt)
            else:
                missed_points.append(pt)

        # Simple score ratio calculation
        total_pts = len(question.expected_points)
        ratio = len(matched_points) / total_pts if total_pts > 0 else 0.5
        score = int(ratio * 10)
        score = max(3, min(10, score))  # Give baseline grade for participation

        # Compose placeholder strengths/weaknesses
        strengths = [f"Mentioned context related to: {p}" for p in matched_points]
        if not strengths:
            strengths = ["Submitted a relevant response covering the prompt."]

        weaknesses = [f"Missed detailed discussion on: {p}" for p in missed_points]
        if not weaknesses:
            weaknesses = ["No critical knowledge gaps identified for the basic parameters."]

        return InterviewFeedback(
            question_id=question.question_id,
            score=score,
            strengths=strengths,
            weaknesses=weaknesses,
            suggested_improvements=[f"Elaborate more on: {', '.join(missed_points)} in future responses."],
            model_answer=f"An ideal answer would systematically cover {', '.join(question.expected_points)}. Example: 'Regarding {question.target_skill}, we apply STAR formats, structuring the performance metrics and scaling indicators.'"
        )

    async def generate_questions(
        self, 
        resume_skills: List[str], 
        target_role: str = "Software Engineer", 
        limit: int = 5
    ) -> List[InterviewQuestion]:
        """
        Generates a custom list of behavioral and technical questions tailored to skills.
        
        Designed to be registered as an MCP tool: 'interview_agent_generate_questions'.

        Args:
            resume_skills: List of skills from the candidate's resume.
            target_role: Target career role being interviewed for.
            limit: Number of questions to generate.

        Returns:
            A list of InterviewQuestion objects.
        """
        if self._client:
            try:
                system_instruction = (
                    "You are a Principal Engineering Manager and Talent Assessor.\n"
                    f"Generate exactly {limit} interview questions for a candidate applying to a {target_role} position.\n\n"
                    "Adhere to the InterviewQuestion schema requirements:\n"
                    "1. Tailor the topics based on these candidate skills: " + ", ".join(resume_skills) + ".\n"
                    "2. Balance the set: include at least 1 behavioral question, 1 system design, and the rest technical coding/framework questions.\n"
                    "3. For each question, specify a unique question_id (sequential integers starting at 1), type, target_skill, difficulty, "
                    "and a list of expected key points ('expected_points') the candidate's answer should ideally hit."
                )

                class QuestionList(BaseModel):
                    questions: List[InterviewQuestion]

                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=f"Generate {limit} questions for the role: {target_role}",
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=QuestionList,
                        temperature=0.4
                    )
                )

                if response.text:
                    data = QuestionList.model_validate_json(response.text)
                    return data.questions
            except Exception:
                # Log execution error internally and drop back to local rules
                pass

        # Fallback to local questions list
        return self._generate_fallback_questions(resume_skills, target_role, limit)

    async def evaluate_answer(
        self, 
        question: InterviewQuestion, 
        user_answer: str
    ) -> InterviewFeedback:
        """
        Grades a single candidate response against the question parameters.
        
        Designed to be registered as an MCP tool: 'interview_agent_evaluate_answer'.

        Args:
            question: The InterviewQuestion object being answered.
            user_answer: The candidate's text response.

        Returns:
            InterviewFeedback containing score, strengths, weaknesses, and a suggested model answer.
        """
        if self._client:
            try:
                system_instruction = (
                    "You are a senior tech interviewer conducting a mock screen. "
                    "Grade the user's answer out of 10. Be constructive but maintain high standards.\n\n"
                    "Adhere to the InterviewFeedback schema:\n"
                    "1. Score (0-10): Genuinely reflect correctness, code quality, and depth of explanation.\n"
                    "2. Strengths: List key terms, concepts, or accurate details they successfully mentioned.\n"
                    "3. Weaknesses: List omissions, errors, or areas where they lacked explanation.\n"
                    "4. Suggested improvements: Outline how they can re-frame or enhance the explanation.\n"
                    "5. Model Answer: Write a concise, exemplary response that hits all expected targets."
                )

                input_prompt = (
                    f"QUESTION: {question.question_text}\n"
                    f"TARGET SKILL: {question.target_skill}\n"
                    f"EXPECTED CRITERIA: {', '.join(question.expected_points)}\n"
                    f"USER ANSWER: {user_answer}"
                )

                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=input_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=InterviewFeedback,
                        temperature=0.2
                    )
                )

                if response.text:
                    return InterviewFeedback.model_validate_json(response.text)
            except Exception:
                # Log execution error internally and drop back to local rules
                pass

        # Fallback to local regex-based score metrics
        return self._evaluate_answer_locally(question, user_answer)

    def compile_session_report(self, session: InterviewSession) -> InterviewSession:
        """
        Aggregates individual question scores and compiles the final interview feedback report.

        Args:
            session: The populated InterviewSession.

        Returns:
            An updated InterviewSession with overall scores and coaching summary.
        """
        if not session.feedback_by_question:
            session.overall_score = 0.0
            session.overall_feedback = "No questions were answered or evaluated in this session."
            session.is_completed = True
            return session

        # Calculate average score (scaled to 100)
        total_score = sum(feed.score for feed in session.feedback_by_question.values())
        avg_score = total_score / len(session.feedback_by_question)
        session.overall_score = float(round(avg_score * 10, 1))

        if session.overall_score >= 80.0:
            rating = "Excellent performance!"
            advice = "You demonstrated strong technical foundations and clear conceptual formatting. Focus on maintaining this depth of explanation."
        elif session.overall_score >= 60.0:
            rating = "Good baseline performance with room for optimization."
            advice = "Your logic is correct, but you could expand on specific architectural details and trade-offs. Spend time articulating implementation specifics."
        else:
            rating = "Further preparation recommended."
            advice = "Review the model answers provided for each question. Focus on incorporating action verbs, quantitative metrics, and core technology details."

        session.overall_feedback = (
            f"Overall Mock Screen Result: {rating}\n"
            f"Candidate achieved an overall score of {session.overall_score}%. {advice}"
        )
        session.is_completed = True
        return session

    def to_adk_agent(self) -> Any:
        """
        Wraps and registers this InterviewAgent instance configuration as a Google ADK Agent.

        Returns:
            An instance of google.adk.agents.Agent.

        Raises:
            ImportError: If the google-adk library is not installed.
        """
        if not HAS_ADK:
            raise ImportError(
                "google-adk package is not installed. Ensure `google-adk` is present in dependencies "
                "to register the ADK agent."
            )

        return AdkAgent(
            name="interview_agent",
            model=self.model_name,
            instruction=(
                "You are the specialist Interview Agent for CareerPilot-AI. "
                "Your role is to formulate technical and behavioral mock interview questions, "
                "grade candidate answers turn-by-turn, compile coaching feedback, and "
                "produce aggregated scoring reports in JSON."
            ),
            tools=[self.generate_questions, self.evaluate_answer]
        )

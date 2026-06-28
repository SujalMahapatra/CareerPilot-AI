"""
FastAPI Main Application for CareerPilot-AI.

This module exposes the core specialized agent functionalities as clean, type-safe REST API endpoints.
It uses FastAPI dependency injection and incorporates proper error handling.
"""

import logging
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("careerpilot_api")

# Core specialized agent imports
from backend.agents.coordinator.coordinator import CoordinatorAgent, RoutingDecision
from backend.agents.resume_agent.resume_agent import ResumeAgent, ResumeAnalysis
from backend.agents.skill_gap_agent.skill_gap_agent import SkillGapAgent, SkillGapAnalysis
from backend.agents.roadmap_agent.roadmap_agent import RoadmapAgent, RoadmapAnalysis
from backend.agents.interview_agent.interview_agent import (
    InterviewAgent,
    InterviewQuestion,
    InterviewFeedback,
    InterviewSession,
)

app = FastAPI(
    title="CareerPilot-AI API",
    description="FastAPI service for the multi-agent career platform CareerPilot-AI.",
    version="1.0.0",
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Dependency Injection Functions ---

def get_coordinator_agent() -> CoordinatorAgent:
    """Dependency provider for CoordinatorAgent."""
    return CoordinatorAgent()

def get_resume_agent() -> ResumeAgent:
    """Dependency provider for ResumeAgent."""
    return ResumeAgent()

def get_skill_gap_agent() -> SkillGapAgent:
    """Dependency provider for SkillGapAgent."""
    return SkillGapAgent()

def get_roadmap_agent() -> RoadmapAgent:
    """Dependency provider for RoadmapAgent."""
    return RoadmapAgent()

def get_interview_agent() -> InterviewAgent:
    """Dependency provider for InterviewAgent."""
    return InterviewAgent()

# --- Request Models ---

class RouteRequest(BaseModel):
    query: str = Field(..., description="The user query to route", example="I need help updating my resume for a Software Engineer role")

class ResumeAnalysisRequest(BaseModel):
    resume_text: str = Field(..., description="Raw text extracted from the candidate's resume")
    job_description: Optional[str] = Field(None, description="Optional target job description to match against")

class SkillGapRequest(BaseModel):
    candidate_skills: List[str] = Field(..., description="List of technical and soft skills possessed by the candidate")
    job_description: str = Field(..., description="Raw text of the target job description")

class RoadmapRequest(BaseModel):
    missing_skills: List[str] = Field(..., description="List of skills currently missing from the candidate's profile")
    timeline_weeks: int = Field(8, description="Desired timeline in weeks (4, 8, or 12)")
    target_role: str = Field("Software Engineer", description="Target job title or role description")

class InterviewQuestionsRequest(BaseModel):
    resume_skills: List[str] = Field(..., description="List of candidate skills extracted from the resume")
    target_role: str = Field("Software Engineer", description="Target career role being interviewed for")
    limit: int = Field(5, description="Number of questions to generate")

class InterviewEvaluateRequest(BaseModel):
    question: InterviewQuestion = Field(..., description="The interview question being answered")
    user_answer: str = Field(..., description="The candidate's text response")

class InterviewReportRequest(BaseModel):
    session: InterviewSession = Field(..., description="The active interview session containing questions and answers")

# --- Endpoints ---

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint to verify API service status.
    """
    return {
        "status": "healthy",
        "service": "CareerPilot-AI API"
    }

@app.post("/api/coordinator/route", response_model=RoutingDecision)
async def route_user_query(
    request: RouteRequest,
    agent: CoordinatorAgent = Depends(get_coordinator_agent)
):
    """
    Analyze user intent and route the query to the correct specialist agent.
    """
    try:
        logger.info(f"Routing query: {request.query}")
        return await agent.route_query(request.query)
    except Exception as e:
        logger.error(f"Error in coordinator routing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Coordinator Agent routing failed: {str(e)}"
        )

@app.post("/api/resume/analyze", response_model=ResumeAnalysis)
async def analyze_resume(
    request: ResumeAnalysisRequest,
    agent: ResumeAgent = Depends(get_resume_agent)
):
    """
    Parse a resume, calculate ATS metrics, extract skills, and generate optimization suggestions.
    """
    try:
        logger.info("Analyzing resume...")
        return await agent.analyze_resume(request.resume_text, request.job_description)
    except Exception as e:
        logger.error(f"Error in resume analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Resume Agent analysis failed: {str(e)}"
        )

@app.post("/api/skill-gap/analyze", response_model=SkillGapAnalysis)
async def analyze_skill_gap(
    request: SkillGapRequest,
    agent: SkillGapAgent = Depends(get_skill_gap_agent)
):
    """
    Compare candidate skills against a job description to find matches, missing skills, and partial matches.
    """
    try:
        logger.info("Analyzing skill gap...")
        return await agent.analyze_skill_gap(request.candidate_skills, request.job_description)
    except Exception as e:
        logger.error(f"Error in skill gap analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Skill Gap Agent analysis failed: {str(e)}"
        )

@app.post("/api/roadmap/generate", response_model=RoadmapAnalysis)
async def generate_roadmap(
    request: RoadmapRequest,
    agent: RoadmapAgent = Depends(get_roadmap_agent)
):
    """
    Generate a chronological learning roadmap to bridge identified skill gaps.
    """
    try:
        logger.info(f"Generating learning roadmap for timeline: {request.timeline_weeks} weeks")
        return await agent.generate_roadmap(
            missing_skills=request.missing_skills,
            timeline_weeks=request.timeline_weeks,
            target_role=request.target_role
        )
    except Exception as e:
        logger.error(f"Error in roadmap generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Roadmap Agent generation failed: {str(e)}"
        )

@app.post("/api/interview/questions", response_model=List[InterviewQuestion])
async def generate_interview_questions(
    request: InterviewQuestionsRequest,
    agent: InterviewAgent = Depends(get_interview_agent)
):
    """
    Generate tailored mock interview questions based on candidate skills and target role.
    """
    try:
        logger.info(f"Generating {request.limit} interview questions for role: {request.target_role}")
        return await agent.generate_questions(
            resume_skills=request.resume_skills,
            target_role=request.target_role,
            limit=request.limit
        )
    except Exception as e:
        logger.error(f"Error in interview questions generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Interview Agent questions generation failed: {str(e)}"
        )

@app.post("/api/interview/evaluate", response_model=InterviewFeedback)
async def evaluate_interview_answer(
    request: InterviewEvaluateRequest,
    agent: InterviewAgent = Depends(get_interview_agent)
):
    """
    Grade a single mock interview response and provide strengths, weaknesses, and a model answer.
    """
    try:
        logger.info(f"Evaluating answer for question ID: {request.question.question_id}")
        return await agent.evaluate_answer(
            question=request.question,
            user_answer=request.user_answer
        )
    except Exception as e:
        logger.error(f"Error in interview answer evaluation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Interview Agent answer evaluation failed: {str(e)}"
        )

@app.post("/api/interview/report", response_model=InterviewSession)
async def compile_interview_report(
    request: InterviewReportRequest,
    agent: InterviewAgent = Depends(get_interview_agent)
):
    """
    Compile the final interview session report, aggregating scores and creating coaching feedback.
    """
    try:
        logger.info(f"Compiling interview session report for session ID: {request.session.session_id}")
        return agent.compile_session_report(request.session)
    except Exception as e:
        logger.error(f"Error in interview report compilation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Interview Agent report compilation failed: {str(e)}"
        )

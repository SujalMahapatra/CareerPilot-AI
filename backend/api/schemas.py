"""
Pydantic Schemas for CareerPilot-AI API.

This module defines the shared request and response models used by the API endpoints.
It imports and re-exposes response models from the specialized agents for unified access.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

# Re-expose existing agent response/data models for unified schema usage
from backend.agents.coordinator.coordinator import RoutingDecision
from backend.agents.resume_agent.resume_agent import (
    ResumeAnalysis,
    ResumeProfile,
    ContactInfo,
    WorkExperience,
    Education,
    Project,
    Suggestion,
    ATSMetrics,
)
from backend.agents.skill_gap_agent.skill_gap_agent import (
    SkillGapAnalysis,
    MatchingSkill,
    MissingSkill,
    PartialMatch,
)
from backend.agents.roadmap_agent.roadmap_agent import (
    RoadmapAnalysis,
    RoadmapPhase,
    LearningResource,
)
from backend.agents.interview_agent.interview_agent import (
    InterviewQuestion,
    InterviewFeedback,
    InterviewSession,
)


# --- Request Models ---

class RouteRequest(BaseModel):
    """Request model for the Coordinator Agent routing endpoint."""
    
    query: str = Field(
        ...,
        description="The user query to route",
        example="I need help updating my resume for a Software Engineer role"
    )


class ResumeAnalysisRequest(BaseModel):
    """Request model for the Resume analysis endpoint."""
    
    resume_text: str = Field(
        ...,
        description="Raw text extracted from the candidate's resume"
    )
    job_description: Optional[str] = Field(
        None,
        description="Optional target job description to match against"
    )


class SkillGapRequest(BaseModel):
    """Request model for the Skill gap analysis endpoint."""
    
    candidate_skills: List[str] = Field(
        ...,
        description="List of technical and soft skills possessed by the candidate"
    )
    job_description: str = Field(
        ...,
        description="Raw text of the target job description"
    )


class RoadmapRequest(BaseModel):
    """Request model for the Learning Roadmap generation endpoint."""
    
    missing_skills: List[str] = Field(
        ...,
        description="List of skills currently missing from the candidate's profile"
    )
    timeline_weeks: int = Field(
        8,
        description="Desired timeline in weeks (4, 8, or 12)"
    )
    target_role: str = Field(
        "Software Engineer",
        description="Target job title or role description"
    )


class InterviewRequest(BaseModel):
    """Request model for generating mock interview questions."""
    
    resume_skills: List[str] = Field(
        ...,
        description="List of candidate skills extracted from the resume"
    )
    target_role: str = Field(
        "Software Engineer",
        description="Target career role being interviewed for"
    )
    limit: int = Field(
        5,
        description="Number of questions to generate"
    )


class InterviewEvaluateRequest(BaseModel):
    """Request model for grading a single mock interview response."""
    
    question: InterviewQuestion = Field(
        ...,
        description="The interview question being answered"
    )
    user_answer: str = Field(
        ...,
        description="The candidate's text response"
    )


class InterviewReportRequest(BaseModel):
    """Request model for compiling the final interview session report."""
    
    session: InterviewSession = Field(
        ...,
        description="The active interview session containing questions and answers"
    )

"""
Skill Gap Agent Module for CareerPilot-AI.

This module contains the SkillGapAgent class along with Pydantic schemas for
skill gap analysis results. The SkillGapAgent evaluates candidate skills against
job description requirements, categorizes them into matched, missing, or partially matching,
and calculates the overall matching percentage.

Compatible with Google ADK framework and designed to act as an MCP-compatible tool.
"""

import os
import re
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

class MatchingSkill(BaseModel):
    """Schema representing a skill successfully matched between candidate and role."""
    
    skill_name: str = Field(..., description="Name of the matching skill")
    proficiency_level: str = Field(
        "intermediate", 
        description="Estimated proficiency level, e.g., 'beginner', 'intermediate', 'advanced'"
    )
    evidence_in_resume: str = Field(
        ..., 
        description="Evidence or context showing where/how the skill was demonstrated in the resume"
    )


class MissingSkill(BaseModel):
    """Schema representing a critical skill missing from the candidate's profile."""
    
    skill_name: str = Field(..., description="Name of the missing skill")
    importance: str = Field(
        "high", 
        description="Importance for the target role: 'critical', 'high', 'medium', 'low'"
    )
    description: str = Field(
        ..., 
        description="Explanation of why this skill is important for the target job description"
    )


class PartialMatch(BaseModel):
    """Schema representing a skill that is partially matched or has transferable experience."""
    
    required_skill: str = Field(..., description="The exact skill required by the employer")
    candidate_skills: List[str] = Field(
        ..., 
        description="Related or overlapping skills the candidate possesses (e.g., MySQL instead of PostgreSQL)"
    )
    gap_description: str = Field(
        ..., 
        description="A description of the gap between what is required and what the candidate knows"
    )
    recommendation: str = Field(
        ..., 
        description="Actionable advice on how to bridge the partial gap"
    )


class SkillGapAnalysis(BaseModel):
    """Payload representing the full output of a skill gap assessment."""
    
    matching_skills: List[MatchingSkill] = Field(default_factory=list)
    missing_skills: List[MissingSkill] = Field(default_factory=list)
    partial_matches: List[PartialMatch] = Field(default_factory=list)
    match_percentage: float = Field(
        ..., 
        description="Calculated match score out of 100", 
        ge=0.0, 
        le=100.0
    )
    recommendations: List[str] = Field(
        default_factory=list, 
        description="General high-level advice on bridging the overall skill gaps"
    )


# =====================================================================
# 2. SkillGapAgent Class
# =====================================================================

class SkillGapAgent:
    """
    Skill Gap Agent specialized in comparing candidate capabilities with target jobs,
    diagnosing missing skills, mapping transferability, and scoring alignment.
    
    Compatible with Google ADK framework and designed to act as an MCP-compatible tool.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        """
        Initializes the SkillGapAgent.

        Args:
            model_name: The Gemini model name used for routing and reasoning.
            api_key: Optional Gemini API key. Defaults to environment variable.
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._client = None

        if HAS_GENAI and self.api_key:
            self._client = genai.Client(api_key=self.api_key)

    def _extract_skills_locally(self, text: str) -> List[str]:
        """Local regex search helper for common technical terms."""
        common_skills = [
            "python", "javascript", "typescript", "java", "c++", "c#", "go", "golang", "rust",
            "html", "css", "react", "angular", "vue", "next.js", "node.js", "express", "fastapi", "django",
            "sql", "postgresql", "mysql", "mongodb", "redis", "firebase", "sqlite",
            "aws", "azure", "gcp", "docker", "kubernetes", "git", "github", "ci/cd", "jenkins",
            "machine learning", "ml", "deep learning", "nlp", "llm", "tensorflow", "pytorch", "scikit-learn",
            "agile", "scrum", "project management", "system design", "data structures", "algorithms"
        ]
        
        extracted = []
        text_lower = text.lower()
        
        for skill in common_skills:
            escaped_skill = re.escape(skill)
            pattern = rf"\b{escaped_skill}\b"
            if skill in ["c++", "c#"]:
                pattern = rf"{escaped_skill}"
                
            if re.search(pattern, text_lower):
                # Normalize names
                matched_label = skill
                if skill == "golang":
                    matched_label = "Go"
                elif skill == "fastapi":
                    matched_label = "FastAPI"
                elif skill == "next.js":
                    matched_label = "Next.js"
                else:
                    matched_label = skill.capitalize() if len(skill) > 3 else skill.upper()
                
                extracted.append(matched_label)
                
        return list(set(extracted))

    def _analyze_skill_gap_locally(
        self, 
        candidate_skills: List[str], 
        job_description: str
    ) -> SkillGapAnalysis:
        """
        Fallback matching logic using rule-based tech categorization and token distance checks.
        
        Args:
            candidate_skills: List of candidate skills.
            job_description: Job description text.

        Returns:
            A populated SkillGapAnalysis model.
        """
        required_skills = self._extract_skills_locally(job_description)
        
        # Skill groups for partial matching mapping
        skill_groups = [
            {"react", "angular", "vue", "next.js", "html", "css", "javascript", "typescript"},  # Frontend
            {"python", "django", "fastapi", "flask", "node.js", "express", "go", "golang", "java", "spring", "c++", "c#", "rust"},  # Backend/Lang
            {"sql", "postgresql", "mysql", "mongodb", "redis", "firebase", "sqlite", "nosql"},  # DB
            {"aws", "azure", "gcp", "docker", "kubernetes", "git", "github", "ci/cd", "jenkins"},  # Cloud/Devops
            {"machine learning", "ml", "deep learning", "nlp", "llm", "tensorflow", "pytorch", "scikit-learn"}  # AI/ML
        ]

        candidate_lower = [c.lower() for c in candidate_skills]
        required_lower = [r.lower() for r in required_skills]

        matched_list: List[MatchingSkill] = []
        partial_list: List[PartialMatch] = []
        missing_list: List[MissingSkill] = []

        # Iterate over required skills to determine matching status
        for req in required_skills:
            req_l = req.lower()
            if req_l in candidate_lower:
                matched_list.append(
                    MatchingSkill(
                        skill_name=req,
                        proficiency_level="intermediate",
                        evidence_in_resume="Matched directly in candidate skills list."
                    )
                )
            else:
                # Check for partial matches in the same technology groups
                related_possessed = []
                for group in skill_groups:
                    if req_l in group:
                        related_possessed = [c for c in candidate_skills if c.lower() in group]
                        break
                
                if related_possessed:
                    partial_list.append(
                        PartialMatch(
                            required_skill=req,
                            candidate_skills=related_possessed,
                            gap_description=f"Candidate possesses related group skills ({', '.join(related_possessed)}) but lacks exact experience with {req}.",
                            recommendation=f"Leverage conceptual understanding of {related_possessed[0]} to quickly pick up syntax and design patterns for {req}."
                        )
                    )
                else:
                    # Check importance from job description context
                    # If skill appears near words like "required", "must", "essential", mark as high/critical
                    jd_lower = job_description.lower()
                    pattern = rf"(?:required|must|essential|critical|strong\s+knowledge)\s+.*?\b{re.escape(req_l)}\b"
                    is_critical = bool(re.search(pattern, jd_lower))
                    importance = "high" if is_critical else "medium"

                    missing_list.append(
                        MissingSkill(
                            skill_name=req,
                            importance=importance,
                            description=f"Skill '{req}' is explicitly mentioned in job requirements, but candidate profile shows no similar domain keywords."
                        )
                    )

        # Calculate matching percentage
        total_req = len(required_skills)
        if total_req > 0:
            # Matched counts as 1.0, Partial counts as 0.5, Missing counts as 0
            score = (len(matched_list) * 1.0 + len(partial_list) * 0.5) / total_req
            match_pct = float(round(score * 100, 1))
        else:
            match_pct = 100.0 if len(candidate_skills) > 0 else 0.0

        # General recommendations
        recommendations = [
            f"Your skills match approximately {match_pct}% of the job description requirements.",
        ]
        if missing_list:
            recommendations.append(
                f"Prioritize learning {', '.join([m.skill_name for m in missing_list[:3]])} which are core missing skill areas."
            )
        if partial_list:
            recommendations.append(
                "Emphasize transferable skills during discussions to mitigate worries about exact tool familiarity."
            )

        return SkillGapAnalysis(
            matching_skills=matched_list,
            missing_skills=missing_list,
            partial_matches=partial_list,
            match_percentage=match_pct,
            recommendations=recommendations
        )

    async def analyze_skill_gap(
        self, 
        candidate_skills: List[str], 
        job_description: str
    ) -> SkillGapAnalysis:
        """
        Compares candidate skills against target job description requirements.
        
        Designed to be registered as an MCP tool: 'skill_gap_agent_analyze_skill_gap'.

        Args:
            candidate_skills: List of skills extracted from candidate resume.
            job_description: Raw text of the target job description.

        Returns:
            SkillGapAnalysis schema containing lists of match metrics and gap actions.
        """
        if self._client:
            try:
                system_instruction = (
                    "You are a Skill Mapping and Career Architecture Expert.\n"
                    "Analyze the list of candidate skills and compare it against the target job description.\n\n"
                    "Follow these schema-specific rules:\n"
                    "1. Match skills: Classify required skills that candidate has. Detail 'proficiency_level' and 'evidence_in_resume'.\n"
                    "2. Partial matches: Find required skills where the candidate doesn't have the exact skill, "
                    "but has similar/transferable ones (e.g. candidate has Spring Boot but job requires FastAPI). Detail 'gap_description' and 'recommendation'.\n"
                    "3. Missing skills: List key required skills the candidate does not have or has no direct equivalent for. "
                    "Rate 'importance' as 'critical', 'high', 'medium', or 'low'.\n"
                    "4. Match Percentage: Calculate an overall match rating out of 100 based on core requirements alignment.\n"
                    "5. Compile general recommendations on bridging the gaps."
                )

                input_prompt = (
                    f"CANDIDATE SKILLS:\n{', '.join(candidate_skills)}\n\n"
                    f"TARGET JOB DESCRIPTION:\n{job_description}"
                )

                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=input_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=SkillGapAnalysis,
                        temperature=0.1
                    )
                )

                if response.text:
                    return SkillGapAnalysis.model_validate_json(response.text)
            except Exception:
                # Log execution error internally and drop back to local rules
                pass

        # Fallback to local rule-based analysis
        return self._analyze_skill_gap_locally(candidate_skills, job_description)

    def to_adk_agent(self) -> Any:
        """
        Wraps and registers this SkillGapAgent instance configuration as a Google ADK Agent.

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
            name="skill_gap_agent",
            model=self.model_name,
            instruction=(
                "You are the specialist Skill Gap Agent for CareerPilot-AI. "
                "Your role is to map candidate skills to job specifications, locate missing capabilities, "
                "propose transferable skills adjustments, and yield structured match analyses in JSON."
            ),
            tools=[self.analyze_skill_gap]
        )

"""
Roadmap Agent Module for CareerPilot-AI.

This module contains the RoadmapAgent class along with Pydantic schemas for
learning roadmaps. The RoadmapAgent generates targeted learning plans divided into chronological
phases based on identified skill gaps and target timelines (4, 8, or 12 weeks),
recommending learning resources, projects, and certifications.

Compatible with Google ADK framework and designed to act as an MCP-compatible tool.
"""

import os
import math
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

class LearningResource(BaseModel):
    """Schema representing a recommended course, tutorial, book, or doc page."""
    
    name: str = Field(..., description="Title of the resource")
    url: Optional[str] = Field(None, description="Direct link to the resource")
    resource_type: str = Field(
        ..., 
        description="Type of resource: 'course', 'documentation', 'book', 'tutorial', 'video'"
    )
    platform: Optional[str] = Field(None, description="Provider/Platform, e.g., 'Coursera', 'Udemy', 'Official'")
    cost: str = Field("Free", description="Cost status: 'Free', 'Paid', 'Subscription'")


class RoadmapPhase(BaseModel):
    """Schema representing a single chronological phase of the learning path."""
    
    phase_number: int = Field(..., description="Sequential index of the phase, starting at 1")
    title: str = Field(..., description="High-level descriptive title of the phase")
    duration_weeks: int = Field(..., description="Duration of this learning phase in weeks")
    topics_to_learn: List[str] = Field(
        default_factory=list,
        description="Key core concepts and skills covered during this phase"
    )
    resources: List[LearningResource] = Field(
        default_factory=list, 
        description="Associated learning materials and web links"
    )
    recommended_projects: List[str] = Field(
        default_factory=list,
        description="Actionable hands-on projects to build to solidify knowledge in this phase"
    )


class RoadmapAnalysis(BaseModel):
    """Payload representing a complete structured learning roadmap."""
    
    summary: str = Field(..., description="High-level overview and study advice for the roadmap")
    target_role: str = Field(..., description="The career path or job title target")
    timeline_weeks: int = Field(..., description="Total length of the learning plan in weeks")
    phases: List[RoadmapPhase] = Field(default_factory=list)
    certifications: List[str] = Field(
        default_factory=list, 
        description="Industry credentials/certifications that add credibility to this path"
    )


# =====================================================================
# 2. RoadmapAgent Class
# =====================================================================

class RoadmapAgent:
    """
    Roadmap Agent specialized in compiling custom study paths, resource recommendations,
    projects, and certifications, optimized for candidate speed and target timelines.
    
    Compatible with Google ADK framework and designed to act as an MCP-compatible tool.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        """
        Initializes the RoadmapAgent.

        Args:
            model_name: The Gemini model name used for routing and reasoning.
            api_key: Optional Gemini API key. Defaults to environment variable.
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._client = None

        if HAS_GENAI and self.api_key:
            self._client = genai.Client(api_key=self.api_key)

    def _get_preset_resources(self, skill: str) -> List[LearningResource]:
        """Returns pre-defined learning resources for common technical terms."""
        skill_lower = skill.lower()
        resources = []

        if "python" in skill_lower:
            resources.append(LearningResource(
                name="Python Core Official Documentation",
                url="https://docs.python.org/3/",
                resource_type="documentation",
                platform="Official",
                cost="Free"
            ))
            resources.append(LearningResource(
                name="Google IT Automation with Python Professional Certificate",
                url="https://www.coursera.org/professional-certificates/google-it-automation",
                resource_type="course",
                platform="Coursera",
                cost="Subscription"
            ))
        elif "react" in skill_lower:
            resources.append(LearningResource(
                name="React Official Quick Start Guide",
                url="https://react.dev/learn",
                resource_type="documentation",
                platform="Official",
                cost="Free"
            ))
            resources.append(LearningResource(
                name="React - The Complete Guide (incl. Hooks, React Router, Redux)",
                url="https://www.udemy.com/course/react-the-complete-guide-incl-redux/",
                resource_type="course",
                platform="Udemy",
                cost="Paid"
            ))
        elif "sql" in skill_lower or "postgres" in skill_lower:
            resources.append(LearningResource(
                name="PostgreSQL Tutorial for Beginners",
                url="https://www.postgresqltutorial.com/",
                resource_type="tutorial",
                platform="Official",
                cost="Free"
            ))
        elif "aws" in skill_lower:
            resources.append(LearningResource(
                name="AWS Cloud Practitioner Essentials",
                url="https://aws.amazon.com/training/digital/aws-cloud-practitioner-essentials/",
                resource_type="course",
                platform="AWS Training",
                cost="Free"
            ))
        elif "machine learning" in skill_lower or "pytorch" in skill_lower or "tensorflow" in skill_lower:
            resources.append(LearningResource(
                name="Machine Learning Specialization by Andrew Ng",
                url="https://www.coursera.org/specializations/machine-learning-introduction",
                resource_type="course",
                platform="Coursera",
                cost="Subscription"
            ))
            resources.append(LearningResource(
                name="PyTorch Tutorials - Getting Started Guide",
                url="https://pytorch.org/tutorials/",
                resource_type="documentation",
                platform="Official",
                cost="Free"
            ))
        else:
            # Dynamic fallback resource template
            resources.append(LearningResource(
                name=f"Official {skill} Documentation",
                url=None,
                resource_type="documentation",
                platform="Official Docs",
                cost="Free"
            ))
            resources.append(LearningResource(
                name=f"Introduction to {skill} & Best Practices",
                url=None,
                resource_type="tutorial",
                platform="YouTube/Medium",
                cost="Free"
            ))

        return resources

    def _generate_fallback_roadmap(
        self, 
        missing_skills: List[str], 
        timeline_weeks: int, 
        target_role: str
    ) -> RoadmapAnalysis:
        """
        Generates a chronological timeline dividing skills and resources locally.
        
        Args:
            missing_skills: List of skills to study.
            timeline_weeks: Length of the study plan.
            target_role: Target career title.

        Returns:
            RoadmapAnalysis structure.
        """
        # Ensure fallback has at least some placeholder if no missing skills are provided
        skills_to_plan = missing_skills if missing_skills else ["General Industry Standards", "Core Stack Frameworks"]
        
        # Configure phase dimensions based on requested timeline duration
        # We target 2 phases for 4-weeks, 3 for 8-weeks, and 4 for 12-weeks
        if timeline_weeks <= 4:
            num_phases = 2
        elif timeline_weeks <= 8:
            num_phases = 3
        else:
            num_phases = 4

        weeks_per_phase = int(math.ceil(timeline_weeks / num_phases))
        
        # Distribute skills across phases evenly
        phase_skills: List[List[str]] = [[] for _ in range(num_phases)]
        for idx, skill in enumerate(skills_to_plan):
            phase_idx = idx % num_phases
            phase_skills[phase_idx].append(skill)

        phases: List[RoadmapPhase] = []
        for i in range(num_phases):
            p_skills = phase_skills[i]
            if not p_skills:
                p_skills = ["Advanced Practices"]
                
            p_resources: List[LearningResource] = []
            for skill in p_skills:
                p_resources.extend(self._get_preset_resources(skill))

            # Build mock portfolio project names
            project_name = f"Build a {p_skills[0]} Capstone Application"
            if len(p_skills) > 1:
                project_name = f"Integrate {p_skills[0]} with {p_skills[1]} in a full-stack dashboard"

            phases.append(
                RoadmapPhase(
                    phase_number=i + 1,
                    title=f"Phase {i + 1}: Mastering {', '.join(p_skills[:2])}",
                    duration_weeks=weeks_per_phase,
                    topics_to_learn=p_skills,
                    resources=p_resources,
                    recommended_projects=[project_name]
                )
            )

        # Map certifications
        certs = []
        skills_lower = [s.lower() for s in skills_to_plan]
        if any("aws" in s or "cloud" in s for s in skills_lower):
            certs.append("AWS Certified Cloud Practitioner")
        if any("ml" in s or "machine learning" in s for s in skills_lower):
            certs.append("TensorFlow Developer Certificate")
        if not certs:
            certs.append(f"General {target_role} Skill Certification")

        summary = (
            f"This fallback plan is tailored to prepare you for the {target_role} path "
            f"over a {timeline_weeks}-week curriculum. Focus on hands-on project deliverables "
            f"in each milestone to maximize retention."
        )

        return RoadmapAnalysis(
            summary=summary,
            target_role=target_role,
            timeline_weeks=timeline_weeks,
            phases=phases,
            certifications=certs
        )

    async def generate_roadmap(
        self, 
        missing_skills: List[str], 
        timeline_weeks: int = 8, 
        target_role: str = "Software Engineer"
    ) -> RoadmapAnalysis:
        """
        Generates a chronological study path based on skill gap elements.
        
        Designed to be registered as an MCP tool: 'roadmap_agent_generate_roadmap'.

        Args:
            missing_skills: List of skills currently missing from candidate profile.
            timeline_weeks: Desired timeline length in weeks (typically 4, 8, or 12).
            target_role: Target job description title.

        Returns:
            RoadmapAnalysis schema containing phases, resources, and checklists.
        """
        # Enforce timeline standards
        if timeline_weeks not in [4, 8, 12]:
            # Adjust to nearest supported baseline
            if timeline_weeks < 6:
                timeline_weeks = 4
            elif timeline_weeks < 10:
                timeline_weeks = 8
            else:
                timeline_weeks = 12

        if self._client:
            try:
                system_instruction = (
                    "You are a Senior Technical Curriculum Designer and Career Coach.\n"
                    f"Create a structured, highly realistic learning roadmap for a candidate aiming to become a {target_role} "
                    f"in exactly {timeline_weeks} weeks.\n\n"
                    "Adhere to the RoadmapAnalysis schema requirements:\n"
                    "1. Construct chronological RoadmapPhases matching the target duration (e.g. split a 4-week timeline into 2 phases, "
                    "an 8-week timeline into 3 or 4 phases, and a 12-week timeline into 4 phases).\n"
                    "2. Assign specific missing skills to their logical phase (prerequisites first).\n"
                    "3. For each phase, list recommended learning resources (books, docs, videos, courses) and "
                    "recommend concrete, hands-on portfolio projects to build.\n"
                    "4. Add relevant industry certifications to aim for.\n"
                    "5. Keep descriptions precise, actionable, and encouraging."
                )

                input_prompt = (
                    f"TARGET ROLE: {target_role}\n"
                    f"TOTAL TIMELINE WEEKS: {timeline_weeks}\n"
                    f"MISSING SKILLS TO INTEGRATE: {', '.join(missing_skills)}"
                )

                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=input_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=RoadmapAnalysis,
                        temperature=0.2
                    )
                )

                if response.text:
                    return RoadmapAnalysis.model_validate_json(response.text)
            except Exception:
                # Log execution error internally and drop back to local rules
                pass

        # Fallback to local rule-based generation
        return self._generate_fallback_roadmap(missing_skills, timeline_weeks, target_role)

    def to_adk_agent(self) -> Any:
        """
        Wraps and registers this RoadmapAgent instance configuration as a Google ADK Agent.

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
            name="roadmap_agent",
            model=self.model_name,
            instruction=(
                "You are the specialist Roadmap Agent for CareerPilot-AI. "
                "Your role is to formulate week-by-week study roadmaps, suggest online courses and certifications, "
                "propose portfolio projects, and output structural study schedules in JSON format."
            ),
            tools=[self.generate_roadmap]
        )

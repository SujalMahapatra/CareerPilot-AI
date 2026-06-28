"""
Model Context Protocol (MCP) Server for CareerPilot-AI.

This module sets up a FastMCP server exposing tools from the specialized agents
(Resume Agent, Skill Gap Agent, Roadmap Agent, and Interview Agent). This enables
external LLMs or systems using MCP to leverage the core functionalities of the CareerPilot-AI platform.

Designed to be run using standard input/output (stdio) transport.
"""

import os
import sys
from typing import List, Optional

# Dynamically add the project root directory to sys.path to allow clean backend imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Graceful import check for FastMCP
try:
    from mcp.server.fastmcp import FastMCP
    HAS_MCP = True
except ImportError:
    HAS_MCP = False

# Import core specialized agents
from backend.agents.resume_agent.resume_agent import ResumeAgent
from backend.agents.skill_gap_agent.skill_gap_agent import SkillGapAgent
from backend.agents.roadmap_agent.roadmap_agent import RoadmapAgent
from backend.agents.interview_agent.interview_agent import InterviewAgent


# Initialize agents
resume_agent = ResumeAgent()
skill_gap_agent = SkillGapAgent()
roadmap_agent = RoadmapAgent()
interview_agent = InterviewAgent()


if HAS_MCP:
    # Instantiate FastMCP server
    mcp = FastMCP("CareerPilot-AI Server")

    # =====================================================================
    # Tool Registrations
    # =====================================================================

    @mcp.tool()
    async def analyze_resume(resume_text: str, job_description: Optional[str] = None) -> str:
        """
        Parse raw resume text, calculate ATS scores, and suggest formatting optimizations.

        Args:
            resume_text: Raw content of the candidate's resume.
            job_description: Optional target job description to match against.

        Returns:
            JSON string representing the ResumeAnalysis model.
        """
        analysis = await resume_agent.analyze_resume(resume_text, job_description)
        return analysis.model_dump_json()


    @mcp.tool()
    def extract_skills(resume_text: str) -> List[str]:
        """
        Extract professional and technical skills from raw resume text.

        Args:
            resume_text: Raw content of the candidate's resume.

        Returns:
            List of unique, standardized skill terms.
        """
        return resume_agent.extract_skills(resume_text)


    @mcp.tool()
    async def analyze_skill_gap(candidate_skills: List[str], job_description: str) -> str:
        """
        Compare candidate skills against job requirements to identify matched, missing, and partial skills.

        Args:
            candidate_skills: List of candidate skills.
            job_description: Raw text of the target job description.

        Returns:
            JSON string representing the SkillGapAnalysis model.
        """
        gap_report = await skill_gap_agent.analyze_skill_gap(candidate_skills, job_description)
        return gap_report.model_dump_json()


    @mcp.tool()
    async def generate_roadmap(
        missing_skills: List[str], 
        timeline_weeks: int = 8, 
        target_role: str = "Software Engineer"
    ) -> str:
        """
        Generate a targeted learning roadmap (4, 8, or 12 weeks) with courses, resources, and projects.

        Args:
            missing_skills: List of skills to cover in the roadmap.
            timeline_weeks: Desired timeline in weeks (typically 4, 8, or 12).
            target_role: Target job title.

        Returns:
            JSON string representing the RoadmapAnalysis model.
        """
        roadmap = await roadmap_agent.generate_roadmap(missing_skills, timeline_weeks, target_role)
        return roadmap.model_dump_json()


    @mcp.tool()
    async def generate_interview_questions(
        resume_skills: List[str], 
        target_role: str = "Software Engineer", 
        limit: int = 5
    ) -> str:
        """
        Generate behavioral and technical interview questions based on candidate skills.

        Args:
            resume_skills: List of candidate skills.
            target_role: Target job title.
            limit: Number of questions to generate.

        Returns:
            JSON string representing a list of InterviewQuestion models.
        """
        questions = await interview_agent.generate_questions(resume_skills, target_role, limit)
        # Format list as a JSON array string
        return f"[{','.join([q.model_dump_json() for q in questions])}]"


else:
    # Mock fallback server wrapper if mcp package is missing in environment
    class MockFastMCP:
        def __init__(self, name: str):
            self.name = name

        def run(self):
            print(f"[{self.name}] Mock server running. Install 'mcp' to enable real Model Context Protocol.")

    mcp = MockFastMCP("CareerPilot-AI Server Fallback")


# =====================================================================
# Server Startup
# =====================================================================

if __name__ == "__main__":
    print("Starting MCP Server...")
    print(f"MCP Available: {HAS_MCP}")
    print(f"Server Type: {type(mcp)}")

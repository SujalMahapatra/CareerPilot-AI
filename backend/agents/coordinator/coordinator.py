"""
Coordinator Agent Module for CareerPilot-AI.

This module contains the CoordinatorAgent class, which acts as the entry point
and orchestrator of the multi-agent career platform. It analyzes user queries,
extracts career-related entities, and routes the request to the appropriate specialist
agent (Resume Agent, Skill Gap Agent, Roadmap Agent, or Interview Agent).

It is built to be clean, typed, and future-ready for Google ADK (Agent Development Kit) integration.
"""

import os
from enum import Enum
from typing import Dict, Any, Optional, List
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
    from google.adk.agents import Agent as AdkAgent
    HAS_ADK = True
except ImportError:
    HAS_ADK = False


class RouteTarget(str, Enum):
    """Supported specialist agent routing destinations."""
    
    RESUME = "resume"
    SKILL_GAP = "skill_gap"
    ROADMAP = "roadmap"
    INTERVIEW = "interview"
    GENERAL = "general"


class RoutingDecision(BaseModel):
    """Schema representing the routing decision and reasoning output by the Coordinator."""
    
    target_agent: RouteTarget = Field(
        ...,
        description="The specialist agent to route the query to: 'resume', 'skill_gap', 'roadmap', 'interview', or 'general'."
    )
    reasoning: str = Field(
        ...,
        description="Detailed logical reasoning explaining why this routing target was selected."
    )
    confidence: float = Field(
        ...,
        description="Confidence score for the routing decision, ranging from 0.0 to 1.0.",
        ge=0.0,
        le=1.0
    )
    extracted_entities: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key entities extracted from the user's query, such as 'job_title', 'skills', 'timeline'."
    )


class CoordinatorAgent:
    """
    Coordinator Agent representing the brain of the CareerPilot-AI platform.
    
    Responsible for analyzing user intent and orchestrating downstream agents.
    It can run standalone using standard Google GenAI client or as an ADK-wrapped agent node.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        """
        Initializes the CoordinatorAgent.

        Args:
            model_name: The Gemini model name to use for routing.
            api_key: Optional Gemini API key. Defaults to environment variable.
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._client = None

        if HAS_GENAI and self.api_key:
            # Initialize modern google-genai Client
            self._client = genai.Client(api_key=self.api_key)

    def _route_by_rules(self, query: str) -> RoutingDecision:
        """
        Fallback keyword-based routing mechanism when GenAI client is uninitialized.
        
        Args:
            query: The user's query.

        Returns:
            A standard RoutingDecision object.
        """
        query_lower = query.lower()

        # Keyword mapping lists
        resume_keywords = ["resume", "cv", "bullet", "ats", "experience", "education", "format", "tailor"]
        skill_gap_keywords = ["skill", "gap", "missing", "required", "compare", "match", "qualifications"]
        roadmap_keywords = ["roadmap", "path", "timeline", "study", "learn", "course", "resource", "plan"]
        interview_keywords = ["interview", "mock", "question", "prep", "technical", "behavioral", "code review"]

        # Calculate keyword match counts
        scores = {
            RouteTarget.RESUME: sum(1 for kw in resume_keywords if kw in query_lower),
            RouteTarget.SKILL_GAP: sum(1 for kw in skill_gap_keywords if kw in query_lower),
            RouteTarget.ROADMAP: sum(1 for kw in roadmap_keywords if kw in query_lower),
            RouteTarget.INTERVIEW: sum(1 for kw in interview_keywords if kw in query_lower),
        }

        # Determine the target with the highest match score
        best_target = max(scores, key=scores.get)
        
        # If no keywords matched, default to general chat/guidance
        if scores[best_target] == 0:
            return RoutingDecision(
                target_agent=RouteTarget.GENERAL,
                reasoning="No specific intent keywords matched. Defaulting to general career chat guidance.",
                confidence=0.5,
                extracted_entities={}
            )

        # Simple entity extraction fallback for common job terms
        entities = {}
        for term in ["software engineer", "data scientist", "product manager", "analyst", "designer"]:
            if term in query_lower:
                entities["job_title"] = term
                break

        return RoutingDecision(
            target_agent=best_target,
            reasoning=f"Identified match for '{best_target.value}' domain via rule-based keyword detection.",
            confidence=0.75,
            extracted_entities=entities
        )

    async def route_query(self, query: str) -> RoutingDecision:
        """
        Analyzes the user input and routes it to the designated agent.

        Args:
            query: The user query string.

        Returns:
            RoutingDecision schema with target agent, reasoning, and metadata.
        """
        if self._client:
            try:
                system_instruction = (
                    "You are the central Coordinator Agent for CareerPilot-AI.\n"
                    "Your job is to analyze the user's career query and decide which specialist agent to route it to.\n\n"
                    "The routing options are:\n"
                    "1. 'resume': For resume uploading, tailoring, optimization, bullet editing, or career profile formatting.\n"
                    "2. 'skill_gap': For analyzing current qualifications against job description requirements to identify gaps.\n"
                    "3. 'roadmap': For designing personalized learning curricula, timelines, study guides, or resource sheets.\n"
                    "4. 'interview': For conducting mock interviews (technical/behavioral), coding exercises, or evaluating answers.\n"
                    "5. 'general': For generic questions, chat greetings, or career advice that doesn't fit the above.\n\n"
                    "Identify and extract key entities where applicable: 'job_title', 'skills_mentioned', 'timeline_weeks'."
                )

                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=query,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=RoutingDecision,
                        temperature=0.1,
                    )
                )

                if response.text:
                    return RoutingDecision.model_validate_json(response.text)
            except Exception as e:
                # Log execution error internally and proceed to rule-based fallback
                pass

        # Fallback if no LLM config or in case of SDK errors
        return self._route_by_rules(query)

    def to_adk_agent(self) -> Any:
        """
        Wraps and registers this class configuration as a Google ADK Agent structure.

        Returns:
            An instance of google.adk.agents.Agent.

        Raises:
            ImportError: If the google-adk library is not installed.
        """
        if not HAS_ADK:
            raise ImportError(
                "google-adk package is not installed. Please add `google-adk` to dependencies "
                "to instantiate the ADK version of this agent."
            )

        return AdkAgent(
            name="coordinator_agent",
            model=self.model_name,
            instruction=(
                "You are the central Coordinator Agent for CareerPilot-AI. "
                "Analyze user requests, route them to appropriate specialist nodes, and "
                "always output a structured JSON schema detailing routing target and reasoning."
            ),
            tools=[self.route_query]
        )

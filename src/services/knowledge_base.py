"""
Cambio Labs organizational knowledge base for augmenting RAG context.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

from sqlalchemy import Column, Integer, String, Text, JSON, DateTime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import structlog

from src.core.database import Base


# initializing structured logger
logger = structlog.get_logger(__name__)


class KnowledgeBaseEntry(Base):
    """
    organizational knowledge base entries
    """
    __tablename__ = "knowledge_base"

    id = Column(Integer, primary_key=True, index=True)
    category = Column(String(100), nullable=False, index=True)
    key = Column(String(200), nullable=False, unique=True, index=True)
    content = Column(Text, nullable=False)
    kb_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class CambioLabsKnowledgeBase:
    """
    structured knowledge base for cambio labs organizational information
    """

    MISSION = """
Cambio Labs empowers underestimated BIPOC youth and adults to build economic mobility through technology education, workforce development, and entrepreneurship. We focus on creating pathways for communities that have been systematically excluded from the tech economy, particularly residents of public housing and low-income neighborhoods in New York City.

Our approach is rooted in real-world outcomes rather than deficit framing. We do not view our students as "underserved" or "at-risk" - they are underestimated changemakers with untapped potential. Through hands-on training, mentorship, and entrepreneurial support, we help them transform their lives and communities.
"""

    PROGRAMS = {
        'Journey Platform': """
Journey is Cambio Labs' flagship AI-powered learning platform that makes technology education accessible and engaging. The platform features:

- Gamified Learning Experience: Students earn Cambio Coins for completing modules, maintaining streaks, and demonstrating mastery
- Sparky AI Tutor: An intelligent chatbot that provides personalized support, answers questions, and adapts to each student's learning style
- No-Code Builder: Students can create real applications without writing code, lowering barriers to entry
- Project-Based Curriculum: Focus on building actual products that students can showcase to employers
- Adaptive Pathways: Content adjusts based on student progress, ensuring everyone can succeed at their own pace

Journey has served 237+ students with a 92% completion rate, significantly higher than typical online learning platforms. Students report increased confidence in technology careers and concrete job opportunities after completing Journey modules.
""",

        'StartUp NYCHA': """
StartUp NYCHA is a business accelerator designed specifically for residents of New York City Housing Authority (NYCHA) developments. The program recognizes that public housing residents have entrepreneurial drive but lack access to capital, networks, and business education.

Program Components:
- 12-Week Accelerator: Intensive curriculum covering business planning, financial management, marketing, and pitch development
- Mentorship Network: Pairing each entrepreneur with experienced business mentors from corporate and startup sectors
- Seed Funding: Micro-grants and connections to capital sources for launching and scaling businesses
- Co-Working Space: Access to professional workspace and networking opportunities
- Technical Training: Integration with Journey platform for technology skills development

The pilot cohort of 30 entrepreneurs has launched businesses in food services, personal care, technology services, and creative industries. Several have achieved revenue positive status and hired employees from their communities. This represents economic mobility not just for individuals but for entire families and neighborhoods.
""",

        'RETI': """
The Renewable Energy Technology Incubator (RETI) prepares individuals for careers in the growing clean energy sector. Developed in partnership with Brooklyn SolarWorks and RETI Center, this program addresses the critical shortage of skilled workers in solar installation and renewable energy systems.

Training Includes:
- Solar Panel Installation: Hands-on training in residential and commercial solar systems
- Electrical Systems: Understanding of electrical work, safety protocols, and code compliance
- Energy Auditing: Assessing buildings for energy efficiency opportunities
- Green Building Practices: Sustainable construction and retrofitting techniques
- Business Development: Path to starting solar installation businesses

Graduates earn industry-recognized certifications (NABCEP, OSHA-30) and have 100% job placement rate with partner companies. Starting salaries range from $45,000-$65,000, representing significant wage increases for participants from low-income backgrounds.
""",

        'Cambio Coding & AI': """
Intensive coding bootcamp focused on practical skills for employment in technology roles. Unlike traditional bootcamps that charge $15,000-$20,000, Cambio Coding is 100% free for participants.

Curriculum Covers:
- Full-Stack Web Development: HTML, CSS, JavaScript, React, Node.js, databases
- Artificial Intelligence Fundamentals: Machine learning basics, working with AI APIs, ethical AI
- Mobile Development: Building iOS and Android applications
- Cloud Technologies: AWS, deployment, serverless architecture
- Professional Skills: Git, agile methodologies, technical communication, portfolio building

The program runs 16 weeks with both full-time immersive and part-time evening/weekend options. Students build 5+ projects for their portfolios and receive career coaching including resume writing, interview preparation, and salary negotiation.

Partnerships with Google, Company Ventures, and CUNY provide additional resources, mentorship, and hiring pipelines.
""",

        'Social Entrepreneurship Incubator': """
Supporting entrepreneurs who are building businesses that create positive social and environmental impact alongside profit. This program is for founders who see business as a tool for community transformation.

Support Provided:
- Business Model Development: Designing ventures that balance profit with social mission
- Impact Measurement: Tracking and communicating social outcomes to investors and stakeholders
- Patient Capital Access: Connecting to impact investors, foundations, and alternative funding sources
- Community Engagement: Building authentic relationships with the communities being served
- Sustainability Planning: Ensuring long-term viability of social ventures

Areas of Focus: Education technology, workforce development, affordable housing, food security, environmental sustainability, financial inclusion.

Ventures in the current cohort include a literacy app for ESL learners, affordable healthy meal delivery service, workforce training for formerly incarcerated individuals, and community solar cooperative.
"""
    }

    TEAM = {
        'Sebastián Martín': """
Founder & Executive Director of Cambio Labs. Sebastián brings experience in education technology, social entrepreneurship, and international development. Previously worked with social ventures in Latin America and led technology education initiatives in underserved communities. Holds degrees in Social Entrepreneurship and International Relations. Fluent in English, Spanish, and Portuguese.
""",

        'Michelle Maluwetig': """
Director of Partnerships & Programs. Michelle oversees strategic partnerships with corporations, foundations, and community organizations. She brings expertise in program design, partnership development, and impact measurement. Previously worked in corporate social responsibility at major technology companies. MBA with focus on social innovation.
""",

        'Richard D-Cal Dacalos': """
Lead Technology Instructor & Curriculum Developer. Richard designs and teaches coding, AI, and technology courses. He brings real-world software engineering experience from startups and enterprise companies. Expert in making complex technical concepts accessible to beginners. Computer Science degree with 10+ years industry experience.
""",

        'Andrej Håkansson': """
Journey Platform Lead Developer & AI Architect. Andrej built the Journey learning platform and Sparky AI tutor. He specializes in machine learning, natural language processing, and educational technology. Previously worked on AI products at technology startups. PhD in Computer Science with focus on AI/ML.
""",

        'Angelo Orciuoli': """
StartUp NYCHA Program Manager. Angelo manages the business accelerator including mentor matching, curriculum delivery, and entrepreneur support. He has entrepreneurship experience and deep connections in NYCHA communities. Passionate about economic justice and community wealth building. Business degree with entrepreneurship concentration.
"""
    }

    STATISTICS = {
        'students_served': '237+',
        'completion_rate': '92%',
        'startup_nycha_entrepreneurs': '30',
        'reti_job_placement_rate': '100%',
        'program_cost_to_students': '$0 (100% free)',
        'average_salary_increase': '$20,000+',
        'founding_year': '2019',
        'headquarters': 'Harlem, New York City',
    }

    PARTNERSHIPS = {
        'Company Ventures': 'Provides mentorship, co-working space, and entrepreneurship resources',
        'CUNY (City University of New York)': 'Academic partnerships, student recruitment, and curriculum collaboration',
        'Blackstone LaunchPad': 'Entrepreneurship support, pitch competitions, and investor connections',
        'Google': 'Technology education resources, cloud credits, and career opportunities',
        'RETI Center': 'Renewable energy training, certifications, and job placement',
        'Brooklyn SolarWorks': 'Solar installation training and employment pipeline',
        'NYCHA (New York City Housing Authority)': 'Program delivery in public housing communities',
        'NYC Opportunity': 'Funding through The People\'s Money participatory budgeting',
    }

    VOICE_GUIDELINES = {
        'preferred_terms': [
            'Use "underestimated" instead of "underserved"',
            'Say "real-world outcomes" not "theoretical knowledge"',
            'Emphasize "economic mobility" over "job training"',
            'Describe students as "changemakers" not "beneficiaries"',
            'Use "BIPOC communities" not "minority communities"',
        ],
        'avoid_deficit_framing': [
            'Do not focus on what communities lack',
            'Do not use language of charity or rescue',
            'Do not describe students as "at-risk" or "disadvantaged"',
            'Do not position Cambio as saviors',
        ],
        'action_oriented_language': [
            'Use active verbs: build, create, launch, transform',
            'Focus on agency: students drive their own success',
            'Emphasize concrete outcomes: jobs, businesses, salaries',
            'Highlight systems change not individual uplift',
        ],
        'authentic_voice': [
            'Write like talking to a colleague, not a funder',
            'Avoid jargon and buzzwords',
            'Tell specific stories with details',
            'Use conversational connectors: "that", "which", "where"',
            'Keep punctuation simple, no em dashes',
        ]
    }

    RECENT_WINS = """
- Won The People's Money grant from NYC Opportunity for Design Fellowship Program and Youth Entrepreneurship Program expansion in Harlem ($150,000)
- Graduated first cohort of StartUp NYCHA with 30 entrepreneurs launching businesses
- Achieved 100% job placement rate for RETI solar workforce training graduates
- Grew Journey platform to 237+ active students with 92% completion rate
- Launched AI tutor "Sparky" with 10,000+ successful student interactions
- Formed partnership with Google for technology education resources and cloud infrastructure
- Secured 3-year commitment from Company Ventures for co-working space and mentorship
"""

    async def initialize_knowledge_base(self, session: AsyncSession) -> None:
        """
        initializing knowledge base with cambio labs information
        """
        try:
            logger.info("initializing_knowledge_base")

            # checking if already initialized
            result = await session.execute(
                select(KnowledgeBaseEntry).limit(1)
            )
            if result.scalar_one_or_none():
                logger.info("knowledge_base_already_initialized")
                return

            entries = []

            # adding mission
            entries.append(KnowledgeBaseEntry(
                category='mission',
                key='organization_mission',
                content=self.MISSION.strip(),
                metadata={'priority': 'high'},
            ))

            # adding programs
            for program_name, description in self.PROGRAMS.items():
                entries.append(KnowledgeBaseEntry(
                    category='programs',
                    key=f'program_{program_name.lower().replace(" ", "_")}',
                    content=description.strip(),
                    metadata={'program_name': program_name},
                ))

            # adding team members
            for member_name, bio in self.TEAM.items():
                entries.append(KnowledgeBaseEntry(
                    category='team',
                    key=f'team_{member_name.lower().replace(" ", "_")}',
                    content=bio.strip(),
                    metadata={'member_name': member_name},
                ))

            # adding statistics
            entries.append(KnowledgeBaseEntry(
                category='statistics',
                key='key_statistics',
                content=str(self.STATISTICS),
                metadata=self.STATISTICS,
            ))

            # adding partnerships
            partnerships_content = '\n\n'.join([
                f"{partner}: {description}"
                for partner, description in self.PARTNERSHIPS.items()
            ])
            entries.append(KnowledgeBaseEntry(
                category='partnerships',
                key='strategic_partnerships',
                content=partnerships_content,
                metadata={'partners': list(self.PARTNERSHIPS.keys())},
            ))

            # adding voice guidelines
            voice_content = '\n\n'.join([
                f"{category.replace('_', ' ').title()}:\n" + '\n'.join(f"- {item}" for item in items)
                for category, items in self.VOICE_GUIDELINES.items()
            ])
            entries.append(KnowledgeBaseEntry(
                category='voice',
                key='writing_guidelines',
                content=voice_content,
                metadata=self.VOICE_GUIDELINES,
            ))

            # adding recent wins
            entries.append(KnowledgeBaseEntry(
                category='achievements',
                key='recent_wins',
                content=self.RECENT_WINS.strip(),
                metadata={'year': 2025},
            ))

            # adding all entries to session
            for entry in entries:
                session.add(entry)

            await session.commit()

            logger.info("knowledge_base_initialized", entries_count=len(entries))

        except Exception as e:
            logger.error("knowledge_base_init_error", error=str(e))
            await session.rollback()
            raise

    async def get_by_category(
        self,
        session: AsyncSession,
        category: str,
    ) -> List[KnowledgeBaseEntry]:
        """
        getting all knowledge base entries for specific category
        """
        result = await session.execute(
            select(KnowledgeBaseEntry)
            .where(KnowledgeBaseEntry.category == category)
            .order_by(KnowledgeBaseEntry.key)
        )
        return result.scalars().all()

    async def get_by_key(
        self,
        session: AsyncSession,
        key: str,
    ) -> Optional[KnowledgeBaseEntry]:
        """
        getting specific knowledge base entry by key
        """
        result = await session.execute(
            select(KnowledgeBaseEntry)
            .where(KnowledgeBaseEntry.key == key)
        )
        return result.scalar_one_or_none()

    async def get_organizational_context(
        self,
        session: AsyncSession,
        include_programs: bool = True,
        include_team: bool = False,
    ) -> str:
        """
        getting formatted organizational context for grant writing
        """
        context_parts = []

        # getting mission
        mission = await self.get_by_key(session, 'organization_mission')
        if mission:
            context_parts.append(f"CAMBIO LABS MISSION:\n{mission.content}\n")

        # getting programs if requested
        if include_programs:
            programs = await self.get_by_category(session, 'programs')
            if programs:
                programs_text = "PROGRAMS:\n\n" + "\n\n".join([
                    f"{entry.metadata.get('program_name', 'Program')}:\n{entry.content}"
                    for entry in programs
                ])
                context_parts.append(programs_text)

        # getting team if requested
        if include_team:
            team = await self.get_by_category(session, 'team')
            if team:
                team_text = "TEAM:\n\n" + "\n\n".join([
                    f"{entry.metadata.get('member_name', 'Team Member')}:\n{entry.content}"
                    for entry in team
                ])
                context_parts.append(team_text)

        # getting achievements
        achievements = await self.get_by_key(session, 'recent_wins')
        if achievements:
            context_parts.append(f"RECENT ACHIEVEMENTS:\n{achievements.content}\n")

        # getting voice guidelines
        voice = await self.get_by_key(session, 'writing_guidelines')
        if voice:
            context_parts.append(f"WRITING GUIDELINES:\n{voice.content}\n")

        return '\n\n===\n\n'.join(context_parts)


# creating singleton instance
_knowledge_base: Optional[CambioLabsKnowledgeBase] = None


def get_knowledge_base() -> CambioLabsKnowledgeBase:
    """
    getting singleton knowledge base instance
    """
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = CambioLabsKnowledgeBase()
    return _knowledge_base

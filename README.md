# GrantLab Multi-Agent System

An experimental multi-agent architecture for automated grant writing (research phase)

## What This Was

This repository contains a multi-agent AI system that was built as an experimental approach to grant writing for Cambio Labs. The system uses LangGraph orchestration with four specialized AI agents, PostgreSQL with pgvector for hybrid search, and comprehensive quality assurance scoring.

While the technology works and demonstrates advanced AI engineering concepts, it turned out to be more complex than what a small nonprofit actually needs. This serves as a research artifact showing what happens when you apply enterprise-scale architecture to a problem that can be solved more simply.

## The Honest Assessment

Cambio Labs is a 2 to 10 person nonprofit in Astoria, NY that empowers underestimated BIPOC youth and adults through technology education and entrepreneurship programs. They apply for 30 to 50 grants annually to fund programs like Journey Platform, StartUp NYCHA, Cambio Solar, and Cambio Coding.

The question this project tried to answer was whether a sophisticated multi-agent system could help them write better grants faster. The answer turned out to be yes technically, but no practically. A small nonprofit does not need four AI agents, hybrid vector search with reranking, semantic caching with Redis, and 26,000 lines of quality assurance code. They need something that works reliably without requiring a DevOps team to maintain it.

That realization led to the cloud approach, which you can find at https://github.com/BasirS/grantlab_cloud_system. That system does 90% of what this multi-agent version does but with 200 lines of code instead of 16,000 and costs $5 per year to run instead of requiring database hosting.

## Architecture Overview

The system was designed with four specialized agents working together through LangGraph orchestration.

The ResearchAgent pulls requirements from RFPs and finds similar past grants from the database. It extracts what funders want, searches through 59 historical grant applications for relevant examples, and gathers organizational context about Cambio Labs programs, metrics, and voice.

The DraftingAgent generates content that maintains Cambio Labs authentic writing style. It takes the research context and uses GPT-4 to write grant sections that sound like the organization actually wrote them, which means avoiding generic nonprofit buzzwords and using specific terminology like "underestimated communities" instead of "underserved populations."

The ReviewAgent checks that generated content does not hallucinate facts or lose authentic voice. It uses RAGAS faithfulness metrics to verify everything traces back to real historical data, checks that all funder requirements get addressed, and generates specific suggestions for improvement.

The IntegrationAgent makes sure all sections work together coherently. It validates that terminology stays consistent across different sections, generates an executive summary that reflects the full application, and produces a unified document with proper formatting.

These agents communicate through a state graph where each agent adds information to a shared state object that flows through the pipeline. LangGraph handles the orchestration, deciding which agent runs when and how information gets passed between them.

## Technical Components

The system includes several sophisticated pieces that work together.

The hybrid search engine combines semantic understanding with keyword matching. Semantic search uses OpenAI embeddings stored in PostgreSQL with pgvector extensions to find conceptually similar content even when exact words differ. Keyword search uses BM25 ranking to catch important terms that semantic search might miss. The system combines both approaches and reranks results to get the most relevant historical examples.

The quality assurance system prevents the kind of hallucinations and voice inconsistencies that make AI-generated content obvious. It calculates a faithfulness score by comparing generated text against source documents, checks voice consistency against Cambio Labs writing guidelines, and runs hallucination detection to catch any facts that cannot be traced to real data. The combined score needs to hit 0.95 or higher, which means 95% confidence that the content is both accurate and authentic.

The cost optimization layer tries to keep API expenses reasonable through aggressive caching. Semantic caching stores embeddings of prompts and checks if similar questions have been answered before. Prompt caching reuses expensive system instructions across multiple requests since OpenAI charges less for cached tokens. Smart model routing sends complex tasks to GPT-4 but uses cheaper models like GPT-4-mini or GPT-3.5-turbo for simpler, more formulaic sections.

The Word export system generates professional documents with proper formatting. It creates cover pages with Cambio Labs branding, adds tables of contents with page numbers, formats sections with appropriate heading styles and body text, and includes optional metadata footers for internal tracking.

## File Structure

The codebase is organized into clear modules that separate concerns.

```
multi_agent/
├── src/
│   ├── main.py                      FastAPI application entry point
│   ├── config.py                     Configuration management
│   ├── api/
│   │   ├── routes/
│   │   │   ├── grants.py            Grant generation endpoints
│   │   │   ├── documents.py         Document upload and management
│   │   │   └── auth.py              Authentication endpoints
│   │   └── models/
│   │       └── schemas.py           Pydantic request/response models
│   ├── core/
│   │   ├── database.py              PostgreSQL connection and models
│   │   ├── vector_store.py          pgvector operations
│   │   ├── embeddings.py            Embedding generation
│   │   └── llm.py                   LLM interaction with fallback
│   └── services/
│       ├── document_ingestion.py    Document processing pipeline
│       ├── rag_engine.py            RAG retrieval logic
│       ├── grant_generator.py       Grant generation orchestration
│       ├── hybrid_search.py         Semantic and keyword search
│       ├── knowledge_base.py        Organizational context
│       └── word_export.py           Professional document formatting
├── requirements.txt                  Python dependencies
├── Dockerfile                        Container configuration
├── docker-compose.yml                Service orchestration
└── init_db.sql                       Database schema
```

## What I Learned

Building this taught me several important lessons about matching technology to actual needs.

The first lesson was that complexity has real costs. This system took 220 hours to build, which at even a modest developer rate of $50 per hour equals $11,000 in development costs. For a nonprofit with 2 to 10 employees, that might exceed what they pay some staff members annually. The system also requires PostgreSQL hosting, Redis caching, and container orchestration, which adds ongoing maintenance burden.

The second lesson involved the 80/20 rule in practice. About 80% of the value comes from basic RAG with good prompting, which you can build in 200 lines of code. The other 20% comes from multi-agent orchestration, hybrid search, quality scoring, and caching, which adds 16,000 lines of complexity. That extra 20% is interesting technically but does not translate to grants that are 20% better or 20% faster to generate.

The third lesson was about operational reality. A sophisticated system that requires database administration, container management, and prompt engineering expertise might work great for a large organization with dedicated tech staff. For a small nonprofit where the executive director writes grants between running programs and managing donor relations, a simple script that just works is infinitely more valuable.

The fourth lesson involved honest assessment of whether technology actually helps. The multi-agent system can generate a complete six-section grant application in 5 to 10 minutes with high quality output. That saves 4 to 7 hours compared to writing from scratch. But you know what else saves 4 to 7 hours? Copying and adapting sections from past successful grants manually, which is what small nonprofits have been doing for decades. The AI makes that process faster and more consistent, but it does not fundamentally change the workflow.

## Cost Analysis

The development costs have already been spent, which represents 220 hours of work across multi-agent architecture, RAG engine implementation, quality assurance, document processing, Word export, and testing.

Running costs break down into API usage and infrastructure. For 30 grants per year, OpenAI API costs come to about $11 per year with aggressive caching bringing that down to $5 to $7. Database hosting for PostgreSQL runs $20 to $30 per month, and Redis caching adds another $10 monthly, totaling $360 to $480 annually.

The time savings are real but need context. Using the system takes 1 to 2 hours per grant including review and editing. Writing from scratch takes 6 to 9 hours. That saves 4 to 7 hours per grant, or 120 to 210 hours annually for 30 grants. At $25 per hour, that equals $3,000 to $5,250 in value.

But the comparison is not quite fair because small nonprofits do not have someone making $25 per hour whose only job is writing grants. Everyone wears multiple hats. The better question is whether the system lets a tiny team focus on what actually matters, which is running programs and serving communities.

## Why This Exists on GitHub

This repository serves as a portfolio piece that demonstrates several things.

It shows exploration and experimentation rather than just presenting polished final products. The journey of trying different approaches, encountering challenges, and making data-driven decisions about what actually works is more valuable than hiding the experiments that did not pan out.

It proves understanding of tradeoffs between different architectural approaches. Being able to explain why a simpler solution works better than a complex one requires having built and tested both options.

It demonstrates breadth of technical skills including multi-agent systems, vector databases, hybrid search, quality assurance frameworks, and production deployment considerations. These are advanced concepts that go beyond basic CRUD applications.

It shows honest assessment of when technology helps and when it gets in the way. Real engineering judgment comes from recognizing that the most sophisticated solution is not always the best solution.

## What Actually Got Deployed

After building and testing this multi-agent system, the decision was made to deploy a simpler cloud-based approach instead. You can find that system at https://github.com/BasirS/grantlab_cloud_system.

The cloud system uses a single enhanced generator with voice validation instead of four specialized agents. It maintains the multi-layer RAG architecture because that genuinely improves output quality, but simplifies the orchestration layer. It runs on Streamlit Cloud instead of requiring database hosting. It achieves 95.8% voice authenticity scores, the same as this multi-agent version, while being dramatically easier to maintain.

For anyone considering building something similar, start simple. Get basic RAG working with good prompts first. Only add complexity if you have specific evidence that the simpler approach is not working. Most of the time, it will work fine.

## Technical Specifications

The system uses OpenAI GPT-4 for complex generation tasks and GPT-4-mini for simpler sections. Embeddings come from text-embedding-3-small, which provides 1536-dimensional vectors that balance quality and cost. The vector database runs on PostgreSQL 15+ with pgvector extension version 0.5+. LangGraph handles multi-agent orchestration. FastAPI provides the async web framework. Authentication uses JWT tokens with configurable expiration.

Document processing handles Word documents through python-docx, PDFs through PyPDF2, and various text encodings. Chunking uses 512-token segments with 50-token overlap to maintain context. Hybrid search combines semantic similarity with BM25 keyword ranking and uses reciprocal rank fusion for combining results.

Quality assurance implements RAGAS faithfulness scoring, custom voice consistency checking, and hallucination detection through fact verification against source documents.

## Running Locally

If you want to explore this system, here is how to get it running. This assumes you have Docker, Docker Compose, and an OpenAI API key.

Clone the repository and navigate into it. Create a .env file with your configuration including database URL, OpenAI API key, JWT secret, and optional monitoring keys for Langfuse.

Start the services using docker-compose which launches PostgreSQL with pgvector, Redis for caching, and the FastAPI application. The API will be available at http://localhost:8000 with interactive documentation at http://localhost:8000/docs.

Initialize the database by running the init_db.sql script which creates the necessary tables and extensions.

Upload historical grant documents through the API endpoints to populate the vector database.

Generate grant sections by sending POST requests to the grants endpoint with RFP requirements and desired sections.

Monitor the system through Langfuse if you configured those keys, which tracks all LLM interactions, costs, and performance metrics.

## Deployment Considerations

This system was designed for containerized deployment but was never actually deployed to production. If you were going to deploy it, here are the considerations.

You would need a PostgreSQL instance with pgvector support, which is available through managed services like AWS RDS, Google Cloud SQL, or Supabase. Redis for caching can run through ElastiCache, Google Memorystore, or Redis Cloud. The FastAPI application containerizes well and can deploy to any platform that runs Docker.

Environment variables need secure storage through AWS Secrets Manager, Google Secret Manager, or similar services. Rate limiting should be configured based on expected usage patterns. Monitoring through Langfuse or similar tools becomes important for tracking costs and quality.

Database backups matter because the vector store contains all the historical knowledge. Regular snapshots and point-in-time recovery help protect against data loss.

## Conclusion

This multi-agent system represents sophisticated AI engineering applied to a real problem. The technology works. The architecture is sound. The code quality is good. But it answers the wrong question.

The right question is not "can we build a complex multi-agent system for grant writing" but "what is the simplest thing that would actually help a small nonprofit write better grants faster." The answer to that question led to the cloud system, which you should use instead of this one.

This repository exists to show the journey, not just the destination. Sometimes the most valuable thing you learn from a project is what not to build next time.

Built by Abdul Basir for Cambio Labs
Version: Research/Experimental
Status: Not recommended for production use
Alternative: https://github.com/BasirS/grantlab_cloud_system

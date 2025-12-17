"""
AutoApply AI - Resume to Jobs Pipeline (Cornell CS Final Build)

A multi-agent LangChain application that:
1. Parses resumes into structured JSON (Agent 1)
2. Critiques resumes with generous feedback (Agent 2)
3. Matches resumes to jobs using EITHER Mock Data OR Live DuckDuckGo Search (Agent 3)
4. Generates personalized cover letters (Agent 4)
5. Generates Interview Prep Guides (Agent 5)

Built for Cornell CS class project.
"""

import streamlit as st
import json
import tempfile
import os
import time
import random
from typing import Dict, List, Any

# LangChain Imports
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchResults

# ──────────────────────────────────────────────────────────────────────────────
# 0. Environment & LLM Setup
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv()  # Load .env file

# Cornell API Setup
os.environ['OPENAI_API_KEY'] = os.getenv("API_KEY", "")
os.environ['OPENAI_BASE_URL'] = 'https://api.ai.it.cornell.edu'

llm = ChatOpenAI(
    model="openai.gpt-4o",
    temperature=0.2,
)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Mock Job Data (10 sample jobs for Demo Mode)
# ──────────────────────────────────────────────────────────────────────────────

MOCK_JOBS = [
    {
        "id": 1,
        "title": "Software Engineer - Backend",
        "company": "TechCorp Inc.",
        "location": "New York, NY (Hybrid)",
        "salary": "$120,000 - $150,000",
        "description": "Build scalable backend services using Python, Django, and PostgreSQL. Work with microservices architecture and cloud deployment on AWS.",
        "requirements": ["Python", "Django", "PostgreSQL", "AWS", "REST APIs", "3+ years experience"],
        "level": "mid"
    },
    {
        "id": 2,
        "title": "Machine Learning Engineer",
        "company": "AI Innovations",
        "location": "San Francisco, CA (Remote)",
        "salary": "$140,000 - $180,000",
        "description": "Develop and deploy ML models for production. Experience with PyTorch, TensorFlow, and MLOps practices required.",
        "requirements": ["Python", "PyTorch", "TensorFlow", "MLOps", "Docker", "5+ years experience"],
        "level": "senior"
    },
    {
        "id": 3,
        "title": "Junior Frontend Developer",
        "company": "WebStart Studios",
        "location": "Austin, TX (On-site)",
        "salary": "$65,000 - $85,000",
        "description": "Join our team building modern web applications with React and TypeScript. Great opportunity for recent graduates.",
        "requirements": ["JavaScript", "React", "TypeScript", "CSS", "HTML", "0-2 years experience"],
        "level": "entry"
    },
    {
        "id": 4,
        "title": "Data Scientist",
        "company": "DataDriven Analytics",
        "location": "Boston, MA (Hybrid)",
        "salary": "$110,000 - $140,000",
        "description": "Analyze large datasets and build predictive models. Strong statistics background and Python proficiency required.",
        "requirements": ["Python", "SQL", "Statistics", "Pandas", "Scikit-learn", "Tableau", "3+ years experience"],
        "level": "mid"
    },
    {
        "id": 5,
        "title": "Full Stack Developer",
        "company": "StartupXYZ",
        "location": "Seattle, WA (Remote)",
        "salary": "$100,000 - $130,000",
        "description": "Build end-to-end features for our SaaS platform. Node.js backend with React frontend.",
        "requirements": ["JavaScript", "Node.js", "React", "MongoDB", "Git", "2+ years experience"],
        "level": "mid"
    },
    {
        "id": 6,
        "title": "DevOps Engineer",
        "company": "CloudScale Systems",
        "location": "Denver, CO (Remote)",
        "salary": "$125,000 - $155,000",
        "description": "Manage CI/CD pipelines and cloud infrastructure. Kubernetes and Terraform expertise needed.",
        "requirements": ["Kubernetes", "Docker", "Terraform", "AWS/GCP", "Linux", "Python", "4+ years experience"],
        "level": "senior"
    },
    {
        "id": 7,
        "title": "Software Engineering Intern",
        "company": "BigTech Corp",
        "location": "Mountain View, CA (On-site)",
        "salary": "$45/hour",
        "description": "Summer internship for CS students. Work on real projects with mentorship from senior engineers.",
        "requirements": ["Python or Java", "Data Structures", "Algorithms", "Currently enrolled in CS program"],
        "level": "student"
    },
    {
        "id": 8,
        "title": "Mobile Developer - iOS",
        "company": "AppWorks Inc.",
        "location": "Chicago, IL (Hybrid)",
        "salary": "$95,000 - $125,000",
        "description": "Build native iOS applications using Swift and SwiftUI. App Store deployment experience preferred.",
        "requirements": ["Swift", "SwiftUI", "iOS SDK", "Xcode", "Git", "2+ years experience"],
        "level": "mid"
    },
    {
        "id": 9,
        "title": "Security Engineer",
        "company": "SecureNet",
        "location": "Washington, DC (On-site)",
        "salary": "$130,000 - $160,000",
        "description": "Protect our infrastructure and applications. Penetration testing and security auditing experience required.",
        "requirements": ["Security certifications", "Python", "Network security", "Penetration testing", "5+ years experience"],
        "level": "senior"
    },
    {
        "id": 10,
        "title": "Backend Engineer - Python",
        "company": "FinTech Solutions",
        "location": "New York, NY (Hybrid)",
        "salary": "$115,000 - $145,000",
        "description": "Build financial data processing systems. Strong Python skills and familiarity with finance domain helpful.",
        "requirements": ["Python", "FastAPI", "PostgreSQL", "Redis", "Docker", "3+ years experience"],
        "level": "mid"
    }
]

# ──────────────────────────────────────────────────────────────────────────────
# 2. ATS Checker (Regex & Rule Based)
# ──────────────────────────────────────────────────────────────────────────────

def check_ats_compatibility(raw_text: str, parsed_resume: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple ATS compatibility checker - analyzes raw text patterns.
    No extra API calls needed.
    """
    issues = []
    warnings = []
    
    lines = raw_text.split('\n')
    non_empty_lines = [l for l in lines if l.strip()]
    
    # 1. Check for possible multi-column layout (lots of short fragmented lines)
    if non_empty_lines:
        short_lines = [l for l in non_empty_lines if 0 < len(l.strip()) < 20]
        if len(short_lines) > len(non_empty_lines) * 0.5:
            issues.append("Possible multi-column layout detected - ATS may scramble text order")
    
    # 2. Check for standard section headers
    text_lower = raw_text.lower()
    if "experience" not in text_lower and "work" not in text_lower:
        issues.append("Missing 'Experience' or 'Work' section header")
    if "education" not in text_lower:
        issues.append("Missing 'Education' section header")
    if "skill" not in text_lower:
        warnings.append("Consider adding a 'Skills' section header for better ATS parsing")
    
    # 3. Check contact info from parsed data
    if not parsed_resume.get('email'):
        issues.append("Missing email address")
    if not parsed_resume.get('phone'):
        warnings.append("Missing phone number")
    
    # 4. Check for links
    links = parsed_resume.get('links', [])
    has_linkedin = any('linkedin' in str(l).lower() for l in links)
    has_github = any('github' in str(l).lower() for l in links)
    
    if not has_linkedin:
        warnings.append("No LinkedIn URL found - recommended for professional networking")
    if not has_github and parsed_resume.get('career_level') in ['student', 'entry']:
        warnings.append("No GitHub URL found - recommended for students/entry-level to showcase code")
    
    # 5. Check for skills keywords (important for ATS keyword matching)
    skills = parsed_resume.get('skills', [])
    if len(skills) < 5:
        warnings.append(f"Only {len(skills)} skills listed - consider adding more relevant keywords")
    
    # 6. Check for quantified achievements (look for numbers/percentages)
    import re
    metrics_pattern = r'\d+%|\d+\+|increased|decreased|improved|reduced|saved|\$\d+'
    has_metrics = bool(re.search(metrics_pattern, raw_text, re.IGNORECASE))
    if not has_metrics:
        warnings.append("No quantified achievements found - add metrics like '40% improvement' or '$50K saved'")
    
    # Calculate ATS score
    ats_score = 100
    ats_score -= len(issues) * 15  # Major issues
    ats_score -= len(warnings) * 5  # Minor warnings
    ats_score = max(0, min(100, ats_score))
    
    return {
        "ats_score": ats_score,
        "ats_compatible": len(issues) == 0,
        "issues": issues,
        "warnings": warnings
    }

# ──────────────────────────────────────────────────────────────────────────────
# 3. Utility: File Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_resume_file(uploaded_file) -> str:
    """Load and extract text from uploaded resume file."""
    file_name = uploaded_file.name
    file_extension = file_name.split('.')[-1].lower() if '.' in file_name else 'txt'
    
    # Save to temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}")
    temp_file.write(uploaded_file.getvalue())
    temp_file.close()
    tmp_file_path = temp_file.name
    
    try:
        # Choose loader based on file type
        if file_extension == "pdf":
            loader = PyPDFLoader(tmp_file_path)
        else:
            loader = TextLoader(tmp_file_path)
        
        documents = loader.load()
        text = "\n".join([doc.page_content for doc in documents])
        return text
    finally:
        # Clean up temp file
        os.remove(tmp_file_path)

# ──────────────────────────────────────────────────────────────────────────────
# AGENT 1: Resume Parser
# ──────────────────────────────────────────────────────────────────────────────

PARSER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert resume parser. Extract ALL information from the resume text provided. Don't miss anything valuable!

Return a valid JSON object with exactly these fields:
{{
    "name": "Full name of the candidate",
    "email": "Email address or null if not found",
    "phone": "Phone number or null if not found",
    "location": "City, State or location if mentioned",
    "links": ["ALL URLs found - LinkedIn, GitHub, portfolio, personal website, etc."],
    "education": [
        {{"degree": "Degree name", "institution": "School name", "year": "Graduation year or 'Present'", "gpa": "GPA if listed", "coursework": "Relevant coursework if listed"}}
    ],
    "experience": [
        {{"title": "Job title", "company": "Company name", "location": "Job location if listed", "duration": "Time period", "months": 0, "description": "Full description - include ALL bullet points and details", "type": "internship|full-time|part-time|research|contract"}}
    ],
    "skills": ["List", "of", "ALL", "technical", "skills", "mentioned"],
    "projects": [
        {{"name": "Project name", "description": "Full description with all details", "technologies": ["tech", "used"]}}
    ],
    "awards_and_certs": ["Any awards, honors, achievements, certifications, or licenses"],
    "activities": ["Clubs, volunteering, extracurriculars"],
    "languages": ["Spoken/written languages if mentioned"],
    "total_months_all_roles": 0,
    "years_of_experience": 0,
    "career_level": "student|entry|mid|senior",
    "domains": ["List of domains like ML, Web Dev, Data Science"],
    "additional_info": ["ANYTHING else valuable on the resume that doesn't fit above - don't lose any information!"]
}}

CRITICAL RULES FOR EXPERIENCE CALCULATION (based on industry standards):

1. "years_of_experience" = ONLY full-time positions count
   - Internships do NOT count toward years of experience (industry standard)
   - Research positions do NOT count
   - Part-time does NOT count
   - This is what employers look at for "requires X years experience"

2. "total_months_all_roles" = raw total of all work (for reference only)
   - Add up all internships, research, full-time, etc.
   - A 3-month summer internship = 3 months
   
3. "career_level" is based on FULL-TIME experience only:
   - "student": Currently enrolled OR only has internships/research (no full-time)
   - "entry": 0-2 years full-time experience
   - "mid": 3-5 years full-time experience  
   - "senior": 6+ years full-time experience

4. For each experience entry, estimate "months" based on duration
   - "May 2024 - Aug 2024" = 4 months
   - "Jun 2023 - Aug 2023" = 3 months

Example: A student with 5 internships totaling 19 months has:
- total_months_all_roles: 19
- years_of_experience: 0 (no full-time roles)
- career_level: "student"

IMPORTANT: Be thorough! Capture EVERYTHING. If something doesn't fit a category, put it in additional_info.
If information is missing, use null or empty arrays.
Return ONLY valid JSON, no markdown formatting or extra text."""),
    ("human", "Parse this resume:\n\n{resume_text}")
])

def parse_resume(resume_text: str) -> Dict[str, Any]:
    """Agent 1: Parse resume text into structured JSON."""
    chain = PARSER_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"resume_text": resume_text})
    
    # Clean up the response (remove markdown code blocks if present)
    result = result.strip()
    if result.startswith("```json"):
        result = result[7:]
    if result.startswith("```"):
        result = result[3:]
    if result.endswith("```"):
        result = result[:-3]
    
    return json.loads(result.strip())

# ──────────────────────────────────────────────────────────────────────────────
# AGENT 2: Resume Critic
# ──────────────────────────────────────────────────────────────────────────────

CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior tech recruiter at a FAANG company reviewing resumes.

TASK: Provide honest, actionable feedback on this resume.

STRICT RULES:

1. NO GENERIC FILLER:
   - Don't suggest "add a summary/objective" (top companies skip these)
   - Don't suggest "add soft skills" (waste of space)
   - Don't tell someone with FAANG internships to "do side projects"

2. RESPECT THEIR LEVEL:
   - Student with TikTok/AWS/Google internships = already strong
   - If they have metrics ("70% improvement"), praise it - don't ask for more

3. BE HONEST:
   - Great resume? Say so. Leave suggestions empty.
   - Don't invent flaws to seem helpful.

4. ONLY FLAG REAL ISSUES:
   - Missing contact info
   - Zero relevant experience
   - Unexplained gaps

OUTPUT (valid JSON only):
{{
    "status": "Ready|Needs Work",
    "readiness_score": 0-100,
    "strengths": ["2-3 specific strengths"],
    "major_issues": ["Critical problems only. Empty if none."],
    "suggestions": ["1-2 tips IF needed. Empty if score 85+."],
    "summary": "One sentence verdict"
}}

SCORING:
- 90-100: FAANG-ready, strong experience, quantified impact
- 80-89: Solid, minor tweaks possible
- 70-79: Good foundation, some gaps
- <70: Needs work

Return ONLY valid JSON."""),
    ("human", "Critique this resume:\n\n{resume_json}")
])

def critique_resume(resume_data: Dict[str, Any]) -> Dict[str, Any]:
    """Agent 2: Analyze resume and provide generous feedback."""
    chain = CRITIC_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"resume_json": json.dumps(resume_data, indent=2)})
    
    # Clean up response
    result = result.strip()
    if result.startswith("```json"):
        result = result[7:]
    if result.startswith("```"):
        result = result[3:]
    if result.endswith("```"):
        result = result[:-3]
    
    return json.loads(result.strip())

# ──────────────────────────────────────────────────────────────────────────────
# AGENT 3A: Mock Job Matcher (Demo Mode)
# ──────────────────────────────────────────────────────────────────────────────

MATCHER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert job matching system. Analyze how well a candidate's resume matches each job posting.

For each job, evaluate:
1. Skill overlap (technical skills match)
2. Experience level alignment
3. Domain relevance
4. Overall fit

Return a valid JSON array with one object per job:
[
    {{
        "job_id": 1,
        "title": "Job Title",
        "company": "Company Name",
        "match_score": 85,
        "reasoning": "Brief explanation of why this is/isn't a good match",
        "skill_matches": ["Skills the candidate has that match"],
        "skill_gaps": ["Skills the job requires that candidate lacks"],
        "recommendation": "highly_recommended|good_fit|possible|poor_fit"
    }}
]

Scoring guidelines:
- 85-100: Excellent match (highly_recommended)
- 70-84: Good match (good_fit)
- 50-69: Possible fit with some gaps (possible)
- Below 50: Not a good fit (poor_fit)

Consider career level - don't penalize students for not having senior-level experience.
Return ONLY valid JSON array, no markdown formatting or extra text."""),
    ("human", """Match this resume against the jobs:

RESUME:
{resume_json}

JOBS:
{jobs_json}""")
])

def match_jobs(resume_data: Dict[str, Any], jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Agent 3A: Score and rank jobs against the resume."""
    chain = MATCHER_PROMPT | llm | StrOutputParser()
    result = chain.invoke({
        "resume_json": json.dumps(resume_data, indent=2),
        "jobs_json": json.dumps(jobs, indent=2)
    })
    
    # Clean up response
    result = result.strip()
    if result.startswith("```json"):
        result = result[7:]
    if result.startswith("```"):
        result = result[3:]
    if result.endswith("```"):
        result = result[:-3]
    
    matches = json.loads(result.strip())
    # Sort by match score descending
    return sorted(matches, key=lambda x: x.get("match_score", 0), reverse=True)

# ──────────────────────────────────────────────────────────────────────────────
# AGENT 3B: Live Web Search Matcher (DuckDuckGo - No blocking!)
# ──────────────────────────────────────────────────────────────────────────────

# Initialize search tool with MORE results per query
search_tool = DuckDuckGoSearchResults(num_results=20)

STRATEGIST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a Master Job Search Strategist. Generate **8 diverse search queries** to find CURRENT, OPEN job listings.

CRITICAL REQUIREMENTS:
1. We want FULL-TIME JOBS, NOT internships (unless user specifically wants internships)
2. Jobs must be CURRENTLY HIRING (add "2024" or "2025" or "hiring now" or "apply now")
3. Each query should target DIFFERENT job boards or approaches
4. Include the job title, location, and key skills

TARGET SITES (vary across queries):
- Direct ATS: greenhouse.io, lever.co, ashbyhq.com, workable.com, jobs.lever.co
- Startup/Tech: wellfound.com, builtin.com, weworkremotely.com, ycombinator.com/jobs
- Big Boards: linkedin.com/jobs, indeed.com, glassdoor.com, ziprecruiter.com

QUERY PATTERNS TO USE (mix these):
- "{role} {location} hiring 2025"
- "{role} jobs {skill} apply now"
- "careers {role} {location} open positions"
- "{company_type} {role} remote jobs 2025"
- "join our team {role} {location}"

User wants: {job_type} positions (NOT internships unless specified)
Role: {role}
Location: {location}
Key Skills: {skills}

Return JSON with exactly 8 queries:
{{ "queries": ["query1", "query2", "query3", "query4", "query5", "query6", "query7", "query8"] }}
"""),
    ("human", "Generate 8 diverse search queries for current job openings.")
])

WEB_MATCHER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a Job Listing Extractor. Extract ALL valid job postings from these search results.

CRITICAL RULES:
1. Extract EVERY job posting you can find - we want QUANTITY
2. Only include {job_type} positions (EXCLUDE internships unless user wants them)
3. Jobs should appear to be CURRENT (2024-2025) and OPEN
4. If a listing looks like a real job posting, INCLUDE IT
5. Extract the actual apply link if visible in the snippet

For each job found, create an entry. Extract as many as possible (aim for 15-30+ jobs).

Return a JSON array:
[
    {{
        "job_id": "unique_id_1",
        "title": "Exact Job Title from listing",
        "company": "Company Name",
        "link": "URL to apply (use the link from snippet)",
        "location": "Location or Remote",
        "match_score": 75,
        "reasoning": "Brief match analysis",
        "skill_matches": ["matching skills from resume"],
        "skill_gaps": ["skills job wants that candidate lacks"],
        "requirements": ["key requirements mentioned"],
        "description": "Job description summary",
        "recommendation": "highly_recommended|good_fit|possible|poor_fit"
    }}
]

Scoring Guide:
- 85-100: Excellent skill match + right level (highly_recommended)
- 70-84: Good match with minor gaps (good_fit)
- 50-69: Partial match (possible)
- Below 50: Poor fit (poor_fit)

IMPORTANT: Return as many jobs as you can extract. More is better!"""),
    ("human", """Candidate Profile: 
- Career Level: {career_level}
- Years Experience: {years_exp}
- Skills: {skills}
- Looking for: {job_type} positions

Search Results to analyze:
{search_results}""")
])

def search_live_jobs(resume_data: Dict[str, Any], role: str, location: str, job_type: str = "full-time"):
    """Agent 3B: Search Live Jobs using DuckDuckGo and rank them."""
    
    career_level = resume_data.get('career_level', 'entry').lower()
    skills_list = resume_data.get('skills', [])[:8]
    
    # 2. Run Strategist to generate search queries
    strat_chain = STRATEGIST_PROMPT | llm | JsonOutputParser()
    
    try:
        strategy = strat_chain.invoke({
            "role": role,
            "location": location,
            "skills": ", ".join(skills_list),
            "job_type": job_type,
        })
        queries = strategy.get('queries', [])
    except Exception as e:
        st.error(f"Error generating search queries: {e}")
        # Fallback to manual queries that work well
        queries = [
            f"{role} {location} jobs hiring 2025",
            f"{role} {location} careers apply now",
            f"{role} remote jobs 2025 hiring",
            f"greenhouse.io {role} {location}",
            f"lever.co {role} jobs open",
            f"wellfound {role} startup jobs",
            f"linkedin jobs {role} {location} 2025",
            f"indeed {role} {location} full time",
        ]
    
    st.write(f"🧠 **Strategist Plan:** Generated {len(queries)} search queries")
    
    # Show queries in debug
    with st.expander("🔍 Search Queries", expanded=False):
        for i, q in enumerate(queries, 1):
            st.write(f"{i}. {q}")
    
    # 3. Scout - Search DuckDuckGo with ALL queries
    raw_results = ""
    total_snippets = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, query in enumerate(queries):
        progress_bar.progress((i + 1) / len(queries))
        status_text.text(f"🔍 Searching ({i+1}/{len(queries)}): {query[:40]}...")
        
        try:
            # Small delay to avoid rate limiting
            if i > 0:
                time.sleep(random.uniform(0.3, 0.8))
            
            results = search_tool.run(query)
            if results and len(results) > 50:  # Got meaningful results
                raw_results += f"\n\n=== QUERY {i+1}: {query} ===\n{results}\n"
                # Count approximate snippets
                snippet_count = results.count("snippet:")
                total_snippets += max(snippet_count, 1)
                st.success(f"✅ Query {i+1}: Found ~{max(snippet_count, 1)} results")
            else:
                st.warning(f"⚠️ Query {i+1}: Few/no results")
            
        except Exception as e:
            st.warning(f"⚠️ Query {i+1} failed: {str(e)[:50]}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    st.info(f"📊 Total: ~{total_snippets} search snippets collected from {len(queries)} queries")
    
    if not raw_results.strip():
        st.error("No search results found. Try different search terms or try again.")
        return []
    
    # Show raw results in debug expander
    with st.expander("🔧 Debug: Raw Search Results", expanded=False):
        st.text(raw_results[:8000] + ("..." if len(raw_results) > 8000 else ""))
    
    # 4. Matcher - Analyze and extract ALL jobs
    st.info("🤖 AI is extracting and ranking job listings...")
    match_chain = WEB_MATCHER_PROMPT | llm | JsonOutputParser()
    
    try:
        matches = match_chain.invoke({
            "career_level": career_level,
            "years_exp": resume_data.get('years_of_experience', 0),
            "skills": ", ".join(skills_list),
            "job_type": job_type,
            "search_results": raw_results[:25000]  # Send more context
        })
        
        if not matches:
            st.warning("No valid job postings extracted. The search results may not have contained job listings. Try different keywords.")
            return []
        
        st.success(f"✅ Extracted {len(matches)} job listings!")
        return sorted(matches, key=lambda x: x.get("match_score", 0), reverse=True)
        
    except Exception as e:
        st.error(f"Error analyzing results: {e}")
        return []
    
    # 4. Matcher - Analyze and rank results
    st.text("🤖 Analyzing results...")
    match_chain = WEB_MATCHER_PROMPT | llm | JsonOutputParser()
    
    try:
        matches = match_chain.invoke({
            "resume_summary": f"Career Level: {career_level}, Years Experience: {resume_data.get('years_of_experience', 0)}, Skills: {', '.join(resume_data.get('skills', [])[:10])}",
            "search_results": raw_results[:15000]  # Limit to avoid token issues
        })
        
        if not matches:
            st.warning("No valid job postings were extracted from the search results. Try different search terms.")
            return []
            
        return sorted(matches, key=lambda x: x.get("match_score", 0), reverse=True)
        
    except Exception as e:
        st.error(f"Error analyzing results: {e}")
        return []

# ──────────────────────────────────────────────────────────────────────────────
# AGENT 4: Cover Letter Generator
# ──────────────────────────────────────────────────────────────────────────────

COVER_LETTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert cover letter writer. Write personalized, compelling cover letters that highlight the candidate's relevant experience.

Guidelines:
- Length: 250-300 words
- Tone: Professional but personable
- Structure: Opening hook, relevant experience, why this company, strong close
- Highlight specific skills that match the job
- Be genuine - avoid generic phrases like "I am excited to apply"
- Reference specific projects or achievements when relevant

Write the cover letter directly - no JSON, no formatting instructions."""),
    ("human", """Write a cover letter for this candidate applying to this job:

CANDIDATE RESUME:
{resume_json}

JOB DETAILS:
Title: {job_title}
Company: {company}
Description: {job_description}
Requirements: {job_requirements}""")
])

def generate_cover_letter(resume_data: Dict[str, Any], job: Dict[str, Any]) -> str:
    """Agent 4: Generate a personalized cover letter."""
    
    # Adapter to handle both Mock Jobs (which have 'requirements' as list) 
    # and Web Jobs (where 'requirements' might be list or text)
    reqs = job.get("requirements", [])
    if isinstance(reqs, list):
        req_str = ", ".join(reqs)
    else:
        req_str = str(reqs)

    # Adapter to handle title keys
    job_title = job.get("title") if job.get("title") else job.get("job_title")

    chain = COVER_LETTER_PROMPT | llm | StrOutputParser()
    result = chain.invoke({
        "resume_json": json.dumps(resume_data, indent=2),
        "job_title": job_title,
        "company": job["company"],
        "job_description": job.get("description", "See job link"),
        "job_requirements": req_str
    })
    return result.strip()

# ──────────────────────────────────────────────────────────────────────────────
# AGENT 5: Interview Coach
# ──────────────────────────────────────────────────────────────────────────────

COACH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an Elite Interview Coach.
    Analyze the BATCH of jobs the user is applying to and their resume.
    
    Task: Create a "Master Prep Guide" that prepares the user for this specific set of opportunities.
    
    Identify:
    1. The most common technical skills required across these jobs.
    2. 3 likely technical interview questions based on those patterns.
    3. 2 likely behavioral questions based on the user's resume weaknesses/gaps.
    4. "Star Power": Which project from their resume is the strongest asset for this batch?
    
    Return clean Markdown.
    """),
    ("human", """
    RESUME: {resume_json}
    
    TARGET JOB BATCH:
    {job_list}
    """)
])

def generate_interview_prep(resume_data, job_list):
    """Agent 5: Generate Interview Prep Guide."""
    # Convert job list to a simple string summary for the LLM
    job_summary = "\n".join([f"- {j.get('title', j.get('job_title'))} at {j.get('company')}" for j in job_list])
    
    chain = COACH_PROMPT | llm | StrOutputParser()
    return chain.invoke({
        "resume_json": json.dumps(resume_data),
        "job_list": job_summary
    })

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AutoApply AI",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AutoApply AI")
st.caption("Resume Parser → Critic → Job Matcher (Mock & Live) → Cover Letter Generator → Interview Coach")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    **AutoApply AI** helps you:
    1. 📝 Parse your resume into structured data
    2. ✅ Get feedback on your resume
    3. 🎯 Find matching jobs (Mock or Live!)
    4. ✉️ Generate personalized cover letters
    5. 🥋 Prep for Interviews
    """)
    
    st.divider()
    
    if st.button("🔄 Start Over"):
        st.session_state.clear()
        st.rerun()
    
    st.divider()
    st.caption("Built with LangChain + Streamlit")

# Initialize session state
if "resume_text" not in st.session_state:
    st.session_state.resume_text = None
if "parsed_resume" not in st.session_state:
    st.session_state.parsed_resume = None
if "ats_check" not in st.session_state:
    st.session_state.ats_check = None
if "critique" not in st.session_state:
    st.session_state.critique = None
if "job_matches" not in st.session_state:
    st.session_state.job_matches = None
if "interview_prep" not in st.session_state:
    st.session_state.interview_prep = None

# ─────────────────────────────────────────────────────────────────────────
# Step 1: File Upload
# ─────────────────────────────────────────────────────────────────────────

st.header("Step 1: Upload Your Resume")

uploaded_file = st.file_uploader(
    "Upload your resume (PDF or TXT)",
    type=["pdf", "txt", "md"],
    help="Supported formats: PDF, TXT, MD"
)

if uploaded_file:
    # Process the file
    if st.session_state.resume_text is None or st.session_state.get("last_file") != uploaded_file.name:
        with st.spinner("Reading resume..."):
            st.session_state.resume_text = load_resume_file(uploaded_file)
            st.session_state.last_file = uploaded_file.name
            # Reset downstream state when new file uploaded
            st.session_state.parsed_resume = None
            st.session_state.ats_check = None
            st.session_state.critique = None
            st.session_state.job_matches = None
            st.session_state.interview_prep = None
    
    # Show preview
    with st.expander("📄 Resume Text Preview", expanded=False):
        st.text(st.session_state.resume_text[:2000] + ("..." if len(st.session_state.resume_text) > 2000 else ""))
    
    # Parse button
    if st.session_state.parsed_resume is None:
        if st.button("🔍 Parse Resume", type="primary"):
            with st.spinner("Parsing resume with AI..."):
                try:
                    st.session_state.parsed_resume = parse_resume(st.session_state.resume_text)
                    # Run ATS check immediately after parsing
                    st.session_state.ats_check = check_ats_compatibility(
                        st.session_state.resume_text, 
                        st.session_state.parsed_resume
                    )
                    st.success("✅ Resume parsed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error parsing resume: {e}")

# ─────────────────────────────────────────────────────────────────────────
# Step 2: Show Parsed Resume + Critique
# ─────────────────────────────────────────────────────────────────────────

if st.session_state.parsed_resume:
    st.divider()
    st.header("Step 2: Parsed Resume Data")
    
    resume = st.session_state.parsed_resume
    
    # Display in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Contact Info")
        st.write(f"**Name:** {resume.get('name', 'N/A')}")
        st.write(f"**Email:** {resume.get('email', 'N/A')}")
        st.write(f"**Phone:** {resume.get('phone', 'N/A')}")
        if resume.get('location'):
            st.write(f"**Location:** {resume.get('location')}")
        
        # Links section
        if resume.get('links'):
            st.subheader("🔗 Links")
            for link in resume.get('links'):
                st.write(f"• {link}")
        
        st.subheader("🎓 Education")
        for edu in resume.get('education', []):
            edu_line = f"• {edu.get('degree', 'N/A')} - {edu.get('institution', 'N/A')} ({edu.get('year', 'N/A')})"
            if edu.get('gpa'):
                edu_line += f" | GPA: {edu.get('gpa')}"
            st.write(edu_line)
            if edu.get('coursework'):
                st.write(f"  *Coursework: {edu.get('coursework')}*")
        
        st.subheader("💼 Experience")
        st.write(f"**Years of Experience (full-time only):** {resume.get('years_of_experience', 0)}")
        st.write(f"**Total Months (all roles):** {resume.get('total_months_all_roles', 'N/A')}")
        st.write(f"**Career Level:** {resume.get('career_level', 'N/A')}")
        for exp in resume.get('experience', []):
            exp_type = f" [{exp.get('type', '')}]" if exp.get('type') else ""
            st.write(f"• **{exp.get('title', 'N/A')}** at {exp.get('company', 'N/A')} ({exp.get('duration', 'N/A')}){exp_type}")
    
    with col2:
        st.subheader("🛠️ Skills")
        skills = resume.get('skills', [])
        if skills:
            st.write(", ".join(skills))
        else:
            st.write("No skills extracted")
        
        st.subheader("🚀 Projects")
        for proj in resume.get('projects', []):
            proj_line = f"• **{proj.get('name', 'N/A')}**: {proj.get('description', 'N/A')}"
            if proj.get('technologies'):
                proj_line += f" [{', '.join(proj.get('technologies'))}]"
            st.write(proj_line)
        
        st.subheader("🎯 Domains")
        domains = resume.get('domains', [])
        if domains:
            st.write(", ".join(domains))
        else:
            st.write("No domains identified")
        
        # New sections
        # Combined awards and certifications
        if resume.get('awards_and_certs'):
            st.subheader("🏆 Awards & Certifications")
            for item in resume.get('awards_and_certs'):
                st.write(f"• {item}")
        
        if resume.get('activities'):
            st.subheader("🎭 Activities")
            st.write(", ".join(resume.get('activities')))
        
        if resume.get('additional_info'):
            st.subheader("➕ Additional Info")
            for info in resume.get('additional_info'):
                st.write(f"• {info}")
    
    # Show raw JSON in expander
    with st.expander("🔧 Raw JSON Data"):
        st.json(resume)
    
    # Critique button
    st.divider()
    st.header("Step 3: Resume Critique")
    
    if st.session_state.critique is None:
        if st.button("✅ Analyze Resume", type="primary"):
            with st.spinner("Analyzing your resume..."):
                try:
                    st.session_state.critique = critique_resume(st.session_state.parsed_resume)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error analyzing resume: {e}")
    
    # Display critique
    if st.session_state.critique:
        critique = st.session_state.critique
        ats = st.session_state.ats_check or {}
        
        # Status banner
        status = critique.get('status', 'Unknown')
        score = critique.get('readiness_score', 0)
        ats_score = ats.get('ats_score', 100)
        
        if status == "Ready":
            st.success(f"🎉 Status: **{status}** | Readiness Score: **{score}/100**")
        else:
            st.warning(f"⚠️ Status: **{status}** | Readiness Score: **{score}/100**")
        
        # ATS Status (from our checker, not the LLM)
        if ats.get('ats_compatible', True):
            st.success(f"✅ ATS Compatible (Score: {ats_score}/100)")
        else:
            st.error(f"❌ ATS Issues Detected (Score: {ats_score}/100)")
        
        # Summary
        st.info(critique.get('summary', 'No summary available'))
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Strengths first
            st.subheader("💪 Strengths")
            strengths = critique.get('strengths', [])
            if strengths:
                for s in strengths:
                    st.write(f"• {s}")
            else:
                st.write("No specific strengths highlighted")
            
            st.subheader("🚨 Major Issues")
            issues = critique.get('major_issues', [])
            if issues:
                for issue in issues:
                    st.write(f"• {issue}")
            else:
                st.write("✅ No major issues found!")
        
        with col2:
            # ATS issues from our checker
            ats_issues = ats.get('issues', [])
            ats_warnings = ats.get('warnings', [])
            
            if ats_issues or ats_warnings:
                st.subheader("🤖 ATS Check")
                for issue in ats_issues:
                    st.write(f"❌ {issue}")
                for warning in ats_warnings:
                    st.write(f"⚠️ {warning}")
            
            st.subheader("💡 Suggestions")
            suggestions = critique.get('suggestions', [])
            if suggestions:
                for suggestion in suggestions:
                    st.write(f"• {suggestion}")
            else:
                st.write("✅ No suggestions - resume looks solid!")

# ─────────────────────────────────────────────────────────────────────────
# Step 4: Job Matching (ENHANCED WITH TABS)
# ─────────────────────────────────────────────────────────────────────────

if st.session_state.parsed_resume:
    st.divider()
    st.header("Step 4: Job Matching")
    
    # --- UI UPDATE: ADD TABS FOR MODE SELECTION ---
    tab1, tab2 = st.tabs(["🎮 Mock Data (Demo Mode)", "🌍 Live Web Search (Agent Mode)"])
    
    # TAB 1: ORIGINAL MOCK LOGIC
    with tab1:
        st.caption("Matches your resume against the static database of 10 mock jobs.")
        if st.button("🎯 Match Mock Jobs", type="primary"):
            with st.spinner("Matching your profile against available jobs..."):
                try:
                    st.session_state.job_matches = match_jobs(st.session_state.parsed_resume, MOCK_JOBS)
                    # Reset prep if jobs change
                    st.session_state.interview_prep = None
                except Exception as e:
                    st.error(f"Error matching jobs: {e}")

    # TAB 2: NEW LIVE SEARCH LOGIC
    with tab2:
        st.caption("Uses AI Agents to search the real web (DuckDuckGo) for CURRENT job openings.")
        
        col_search_1, col_search_2, col_search_3 = st.columns(3)
        target_role = col_search_1.text_input("Target Role", "Software Engineer")
        target_loc = col_search_2.text_input("Location", "Remote")
        job_type = col_search_3.selectbox("Job Type", ["full-time", "internship", "contract", "part-time"], index=0)
        
        st.info("💡 Tip: Be specific with role (e.g., 'Backend Engineer Python' instead of just 'Engineer')")
        
        if st.button("🚀 Search Live Jobs", type="primary"):
            with st.spinner("🔍 Searching across multiple job boards..."):
                try:
                    st.session_state.job_matches = search_live_jobs(
                        st.session_state.parsed_resume, 
                        target_role, 
                        target_loc,
                        job_type
                    )
                    # Reset prep if jobs change
                    st.session_state.interview_prep = None
                except Exception as e:
                    st.error(f"Error searching jobs: {e}")

    # Display Results (Works for BOTH Mock and Live jobs)
    if st.session_state.job_matches:
        st.divider()
        st.subheader(f"Found {len(st.session_state.job_matches)} Matches")
        
        # --- NEW FEATURE: INTERVIEW COACH BUTTON ---
        if st.button("🥋 Generate Interview Prep for these Jobs"):
            with st.spinner("Agent 5 (The Coach) is analyzing the job batch..."):
                try:
                    st.session_state.interview_prep = generate_interview_prep(
                        st.session_state.parsed_resume, 
                        st.session_state.job_matches
                    )
                except Exception as e:
                    st.error(f"Error generating prep: {e}")

        # Display Prep Guide if exists
        if st.session_state.interview_prep:
            with st.expander("🥋 Master Interview Prep Guide (Generated by Agent 5)", expanded=True):
                st.markdown(st.session_state.interview_prep)
        
        # Display Job Cards
        for i, match in enumerate(st.session_state.job_matches):
            job_id = match.get('job_id', i) # Fallback to index if no ID
            
            # Handle slight differences in keys between Mock and Live data
            title = match.get('title') or match.get('job_title')
            
            score = match.get('match_score', 0)
            recommendation = match.get('recommendation', 'unknown')
            
            # Color coding based on recommendation
            if recommendation == "highly_recommended":
                color = "🟢"
            elif recommendation == "good_fit":
                color = "🔵"
            elif recommendation == "possible":
                color = "🟡"
            else:
                color = "🔴"
            
            with st.expander(f"{color} **{title}** at {match['company']} - Score: {score}/100"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Location:** {match.get('location')}")
                    # Only mock jobs have salary usually
                    if match.get('salary'):
                        st.write(f"**Salary:** {match.get('salary')}")
                    
                    st.write(f"**Description:** {match.get('description', 'See link for details')}")
                    
                    # Live jobs might have a link
                    if match.get('link'):
                        st.write(f"**Link:** [Apply Here]({match.get('link')})")
                    
                    # Handle requirements (List vs String)
                    reqs = match.get('requirements', [])
                    if isinstance(reqs, list):
                        st.write(f"**Requirements:** {', '.join(reqs)}")
                    else:
                        st.write(f"**Requirements:** {reqs}")
                
                with col2:
                    st.metric("Match Score", f"{score}/100")
                    st.write(f"**Recommendation:** {recommendation.replace('_', ' ').title()}")
                
                st.divider()
                st.write(f"**Why this match:** {match.get('reasoning', 'N/A')}")
                
                skill_col1, skill_col2 = st.columns(2)
                with skill_col1:
                    st.write("**✅ Matching Skills:**")
                    for skill in match.get('skill_matches', []):
                        st.write(f"  • {skill}")
                with skill_col2:
                    st.write("**❌ Skill Gaps:**")
                    for skill in match.get('skill_gaps', []):
                        st.write(f"  • {skill}")
                
                # Cover letter generation
                st.divider()
                cover_letter_key = f"cover_letter_{i}" # Use index to avoid ID conflicts
                
                if cover_letter_key not in st.session_state:
                    if st.button(f"✉️ Generate Cover Letter", key=f"btn_{i}"):
                        with st.spinner("Generating personalized cover letter..."):
                            try:
                                cover_letter = generate_cover_letter(st.session_state.parsed_resume, match)
                                st.session_state[cover_letter_key] = cover_letter
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error generating cover letter: {e}")
                else:
                    st.subheader("📝 Your Cover Letter")
                    st.markdown(st.session_state[cover_letter_key])
                    
                    # Copy button
                    st.download_button(
                        label="📋 Download Cover Letter",
                        data=st.session_state[cover_letter_key],
                        file_name=f"cover_letter_{match['company'].replace(' ', '_')}.txt",
                        mime="text/plain",
                        key=f"download_{i}"
                    )

# ─────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────

st.divider()
st.caption("AutoApply AI - Built with LangChain, GPT-4o, Selenium, and Streamlit")
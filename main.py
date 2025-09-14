import os
import json
import asyncio
import random
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AI Content Platform", version="3.0.0")

# Configure templates
templates = Jinja2Templates(directory="templates")

# Configure Gemini AI
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=api_key)

# Initialize the model
model = genai.GenerativeModel('gemini-pro')
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key, temperature=0.9)

# Content types and platforms
CONTENT_TYPES = ["Educational", "Entertainment", "How-to Guide", "Review", "News", "Opinion", "Listicle"]
PLATFORMS = ["YouTube", "Blog", "Instagram", "Twitter", "TikTok", "LinkedIn"]

# Content angles and formats for diversity
CONTENT_ANGLES = [
    "Beginner's Guide to", "Advanced Techniques for", "The Ultimate Guide to",
    "5 Surprising Facts About", "Mistakes to Avoid with", "Interview with an Expert on",
    "Case Study:", "Behind the Scenes of", "Future of", "History of"
]

CONTENT_FORMATS = [
    "Step-by-Step Tutorial", "Q&A Session", "Live Demonstration", "Comparison Review",
    "Product Showcase", "Industry Analysis", "Personal Story", "Data-Driven Insights",
    "Collaboration with", "Challenge/Accepted"
]

# Pydantic models
class ContentRequest(BaseModel):
    content_type: str
    platforms: List[str]
    niche: str
    audience: str
    goals: str

class ContentIdea(BaseModel):
    title: str
    description: str
    platform: str
    publish_date: str
    tags: List[str]

# Store user content calendars with request hashing
user_calendars = {}

# Utility functions
def generate_request_hash(request: ContentRequest) -> str:
    """Generate a unique hash for the request to use as a cache key"""
    request_str = f"{request.content_type}_{'_'.join(request.platforms)}_{request.niche}_{request.audience}_{request.goals}"
    return hashlib.md5(request_str.encode()).hexdigest()

def generate_content_calendar(request: ContentRequest, user_id: str = "default", force_new: bool = False) -> List[Dict[str, Any]]:
    """Generate a 30-day content calendar using Gemini AI"""
    
    # Generate a unique hash for this request
    request_hash = generate_request_hash(request)
    cache_key = f"{user_id}_{request_hash}"
    
    # Check if we already have a calendar for this user and request
    if not force_new and cache_key in user_calendars:
        calendar_data = user_calendars[cache_key]
        last_generated = datetime.fromisoformat(calendar_data["last_generated"])
        
        # If we generated a calendar recently, return it (cache for 1 hour)
        if (datetime.now() - last_generated).total_seconds() < 3600:
            return calendar_data["calendar"]
    
    # Create a JSON output parser
    parser = JsonOutputParser()
    
    # Get tomorrow's date for starting the calendar
    tomorrow = datetime.now().date() + timedelta(days=1)
    
    # Create the prompt template with more specific instructions
    prompt = PromptTemplate(
        template="""
        As an expert content strategist, create a diverse 30-day content calendar for a content creator.
        
        CONTENT TYPE: {content_type}
        PLATFORMS: {platforms}
        NICHE/TOPIC: {niche}
        TARGET AUDIENCE: {audience}
        GOALS: {goals}
        
        Generate a diverse content calendar with:
        1. Unique, engaging titles for each day
        2. Brief descriptions (1-2 sentences) that explain the content
        3. Appropriate platform for each content piece (rotate through the selected platforms)
        4. Relevant tags/hashtags (3-5 per piece)
        5. Spread content evenly across 30 days starting from {start_date}
        
        Ensure each day has completely different content. Vary the:
        - Topics within the niche
        - Content formats (tutorials, reviews, interviews, etc.)
        - Angles (beginner vs advanced, problem/solution, etc.)
        
        Return ONLY a valid JSON array without any markdown or extra text. Each item should have:
        - title: string (unique and engaging)
        - description: string
        - platform: string (from: {platforms})
        - publish_date: string (YYYY-MM-DD, starting from {start_date})
        - tags: array of strings (3-5 relevant tags)
        
        Example format:
        [
          {{
            "title": "Beginner's Guide to {niche}: Getting Started",
            "description": "Learn the fundamentals of {niche} and how to get started as a complete beginner.",
            "platform": "YouTube",
            "publish_date": "{start_date}",
            "tags": ["beginner", "tutorial", "guide", "{niche}"]
          }},
          {{
            "title": "Advanced Techniques for Mastering {niche}",
            "description": "Discover professional techniques that will take your {niche} skills to the next level.",
            "platform": "Blog",
            "publish_date": "{next_date}",
            "tags": ["advanced", "techniques", "master", "{niche}"]
          }}
        ]
        """,
        input_variables=["content_type", "platforms", "niche", "audience", "goals"],
        partial_variables={
            "start_date": tomorrow.strftime("%Y-%m-%d"),
            "next_date": (tomorrow + timedelta(days=1)).strftime("%Y-%m-%d")
        }
    )
    
    # Create the chain
    chain = prompt | llm | parser
    
    try:
        # Invoke the chain
        result = chain.invoke({
            "content_type": request.content_type,
            "platforms": ", ".join(request.platforms),
            "niche": request.niche,
            "audience": request.audience,
            "goals": request.goals
        })
        
        # Validate and ensure we have 30 unique items
        if len(result) < 30:
            # If we don't have enough items, supplement with fallback
            fallback_items = generate_fallback_calendar(request)
            # Use AI-generated items first, then supplement with fallback
            result.extend(fallback_items[len(result):])
        elif len(result) > 30:
            # If we have too many, trim to 30
            result = result[:30]
        
        # Store the calendar for future use
        user_calendars[cache_key] = {
            "calendar": result,
            "last_generated": datetime.now().isoformat(),
            "request": request.dict()
        }
        
        return result
        
    except OutputParserException as e:
        print(f"Output parsing error: {e}")
        # Fallback to a default calendar if parsing fails
        return generate_fallback_calendar(request)
    except Exception as e:
        print(f"Error generating content calendar: {e}")
        # Fallback to a default calendar if AI fails
        return generate_fallback_calendar(request)

def generate_fallback_calendar(request: ContentRequest) -> List[Dict[str, Any]]:
    """Generate a fallback content calendar with diverse content"""
    calendar = []
    start_date = datetime.now().date() + timedelta(days=1)  # Start from tomorrow
    
    # Generate diverse content for each day
    for i in range(30):
        publish_date = start_date + timedelta(days=i)
        
        # Select a platform in rotation
        platform_idx = i % len(request.platforms)
        platform = request.platforms[platform_idx]
        
        # Select random content angle and format for diversity
        angle = random.choice(CONTENT_ANGLES)
        content_format = random.choice(CONTENT_FORMATS)
        
        # Create unique content for each day
        title = f"{angle} {request.niche}: {content_format}"
        description = f"A {content_format.lower()} about {request.niche} for {request.audience}. {request.goals}."
        
        calendar.append({
            "title": title,
            "description": description,
            "platform": platform,
            "publish_date": publish_date.strftime("%Y-%m-%d"),
            "tags": [request.niche, request.content_type, platform, angle.lower().replace(" ", "-"), content_format.lower().replace(" ", "-")]
        })
    
    return calendar

def detect_trends(niche: str) -> List[str]:
    """Detect current trends in the given niche"""
    
    prompt = PromptTemplate(
        template="""
        As an AI trend detector, identify the top 5 current trends in the following niche: {niche}
        
        Return ONLY a JSON array of trend names without any additional text.
        
        Example: ["Trend 1", "Trend 2", "Trend 3"]
        """,
        input_variables=["niche"]
    )
    
    # Create the chain
    chain = prompt | llm | JsonOutputParser()
    
    try:
        result = chain.invoke({"niche": niche})
        return result
    except Exception as e:
        print(f"Error detecting trends: {e}")
        return [f"Trend in {niche}", "Industry News", "How-to Guides", "Product Reviews", "Expert Opinions"]

# FastAPI routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request, 
            "content_types": CONTENT_TYPES,
            "platforms": PLATFORMS
        }
    )

@app.post("/generate-calendar")
async def generate_calendar_endpoint(
    request: Request,
    content_type: str = Form(...),
    platforms: List[str] = Form(...),
    niche: str = Form(...),
    audience: str = Form(...),
    goals: str = Form(...),
    force_new: bool = Form(False)  # Add a parameter to force new generation
):
    try:
        content_request = ContentRequest(
            content_type=content_type,
            platforms=platforms,
            niche=niche,
            audience=audience,
            goals=goals
        )
        
        # Generate content calendar
        # Use client IP as user ID to differentiate between users
        user_id = request.client.host if request.client else "default"
        calendar = generate_content_calendar(content_request, user_id, force_new)
        
        # Detect trends
        trends = detect_trends(niche)
        
        return JSONResponse({
            "success": True,
            "calendar": calendar,
            "trends": trends,
            "generated_date": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating content: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/refresh-calendar")
async def refresh_calendar(request: Request):
    """Force refresh of the content calendar for a user"""
    user_id = request.client.host if request.client else "default"
    
    # Remove all cached calendars for this user
    keys_to_remove = [key for key in user_calendars.keys() if key.startswith(user_id)]
    for key in keys_to_remove:
        del user_calendars[key]
        
    return {"status": "refreshed", "user_id": user_id, "removed_entries": len(keys_to_remove)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=True)
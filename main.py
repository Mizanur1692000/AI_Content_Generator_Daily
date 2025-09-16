import os
import json
import asyncio
import random
import hashlib
import logging
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
import aiohttp
import serpapi

# Load environment variables
load_dotenv()

# Configure logging with UTF-8 encoding support
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("content_platform")

# Initialize FastAPI app
app = FastAPI(title="AI Content Platform", version="3.0.0")

# Configure templates
templates = Jinja2Templates(directory="templates")

# Configure APIs
api_key = os.getenv("GEMINI_API_KEY")
serpapi_key = os.getenv("SERPAPI_KEY")
youtube_key = os.getenv("YOUTUBE_API_KEY")

logger.info(f"GEMINI_API_KEY found: {bool(api_key)}")
logger.info(f"SERPAPI_KEY found: {bool(serpapi_key)}")
logger.info(f"YOUTUBE_API_KEY found: {bool(youtube_key)}")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=api_key)

# Initialize the model
model = genai.GenerativeModel('gemini-2.5-pro')
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key, temperature=0.9)
logger.info("Gemini model initialized successfully")

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

# Store debug information
debug_info = {}

# Utility functions
def generate_request_hash(request: ContentRequest) -> str:
    """Generate a unique hash for the request to use as a cache key"""
    request_str = f"{request.content_type}_{'_'.join(request.platforms)}_{request.niche}_{request.audience}_{request.goals}"
    return hashlib.md5(request_str.encode()).hexdigest()

def safe_log_message(message):
    """Safely encode message to avoid Unicode errors in Windows terminal"""
    try:
        return message.encode('utf-8', errors='replace').decode('utf-8')
    except:
        return "Unable to display message due to encoding issues"

async def get_search_trends(niche: str) -> List[str]:
    """Get current search trends using SerpAPI"""
    logger.info(f"Fetching search trends for niche: {niche}")
    trends = []
    debug_data = {"api": "serpapi", "niche": niche, "status": "started", "key_found": bool(serpapi_key)}
    
    if not serpapi_key:
        logger.warning("SERPAPI_KEY not found, using fallback trends")
        debug_data["status"] = "no_api_key"
        debug_data["fallback_trends"] = [f"Trend in {niche}", "Industry News", "How-to Guides", "Product Reviews", "Expert Opinions"]
        debug_info["search_trends"] = debug_data
        return debug_data["fallback_trends"]
    
    try:
        async with aiohttp.ClientSession() as session:
            params = {
                'engine': 'google_trends',
                'q': niche,
                'api_key': serpapi_key,
                'data_type': 'TIMESERIES'
            }
            
            debug_data["request_params"] = {k: v for k, v in params.items() if k != 'api_key'}
            logger.debug(f"SerpAPI request params: {debug_data['request_params']}")
            
            async with session.get('https://serpapi.com/search', params=params) as response:
                debug_data["response_status"] = response.status
                logger.debug(f"SerpAPI response status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    debug_data["response_data"] = data
                    
                    # Extract trending topics from the response
                    if 'interest_over_time' in data and 'timeline_data' in data['interest_over_time']:
                        for item in data['interest_over_time']['timeline_data']:
                            if 'values' in item:
                                for value in item['values']:
                                    if 'query' in value:
                                        trends.append(value['query'])
                    
                    # Also try related queries
                    params['data_type'] = 'RELATED_QUERIES'
                    async with session.get('https://serpapi.com/search', params=params) as rel_response:
                        debug_data["related_status"] = rel_response.status
                        logger.debug(f"SerpAPI related queries status: {rel_response.status}")
                        
                        if rel_response.status == 200:
                            rel_data = await rel_response.json()
                            debug_data["related_data"] = rel_data
                            
                            if 'related_queries' in rel_data:
                                for query in rel_data['related_queries']:
                                    if 'query' in query:
                                        trends.append(query['query'])
                
                # If we don't get enough trends, use fallback
                if len(trends) < 3:
                    fallback = [f"{niche} trends", f"latest {niche} news", f"{niche} tips", f"{niche} tutorials"]
                    trends.extend(fallback)
                    debug_data["fallback_added"] = fallback
                    logger.warning(f"Not enough trends found, added fallback: {fallback}")
                    
                debug_data["status"] = "success"
                debug_data["trends_found"] = len(trends)
                debug_data["trends"] = trends
                debug_info["search_trends"] = debug_data
                
                logger.info(f"Found {len(trends)} search trends for {niche}")
                logger.debug(safe_log_message(f"Search trends: {trends}"))
                
                return list(set(trends))[:10]  # Return unique trends, max 10
    
    except Exception as e:
        debug_data["status"] = "error"
        debug_data["error"] = str(e)
        debug_info["search_trends"] = debug_data
        logger.error(f"Error fetching search trends: {e}")
        return [f"Trend in {niche}", "Industry News", "How-to Guides", "Product Reviews", "Expert Opinions"]

async def get_youtube_trends(niche: str) -> List[str]:
    """Get YouTube trending videos related to the niche"""
    logger.info(f"Fetching YouTube trends for niche: {niche}")
    trends = []
    debug_data = {"api": "youtube", "niche": niche, "status": "started", "key_found": bool(youtube_key)}
    
    if not youtube_key:
        logger.warning("YOUTUBE_API_KEY not found, using fallback trends")
        debug_data["status"] = "no_api_key"
        debug_data["fallback_trends"] = [f"{niche} videos", f"{niche} tutorials", f"{niche} reviews"]
        debug_info["youtube_trends"] = debug_data
        return debug_data["fallback_trends"]
    
    try:
        async with aiohttp.ClientSession() as session:
            # Search for popular videos in the niche
            params = {
                'part': 'snippet',
                'q': niche,
                'type': 'video',
                'order': 'viewCount',
                'maxResults': 10,
                'key': youtube_key
            }
            
            debug_data["request_params"] = params
            logger.debug(f"YouTube API request params: {params}")
            
            async with session.get('https://www.googleapis.com/youtube/v3/search', params=params) as response:
                debug_data["response_status"] = response.status
                logger.debug(f"YouTube API response status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    debug_data["response_data"] = data
                    
                    if 'items' in data:
                        for item in data['items']:
                            if 'snippet' in item and 'title' in item['snippet']:
                                trends.append(item['snippet']['title'])
                
                # If we don't get enough trends, use fallback
                if len(trends) < 3:
                    fallback = [f"{niche} videos", f"{niche} tutorials", f"{niche} reviews"]
                    trends.extend(fallback)
                    debug_data["fallback_added"] = fallback
                    logger.warning(f"Not enough YouTube trends found, added fallback: {fallback}")
                    
                debug_data["status"] = "success"
                debug_data["trends_found"] = len(trends)
                debug_data["trends"] = trends
                debug_info["youtube_trends"] = debug_data
                
                logger.info(f"Found {len(trends)} YouTube trends for {niche}")
                logger.debug(safe_log_message(f"YouTube trends: {trends}"))
                
                return trends[:5]  # Return max 5 trends
    
    except Exception as e:
        debug_data["status"] = "error"
        debug_data["error"] = str(e)
        debug_info["youtube_trends"] = debug_data
        logger.error(f"Error fetching YouTube trends: {e}")
        return [f"{niche} videos", f"{niche} tutorials", f"{niche} reviews"]

async def generate_content_calendar(request: ContentRequest, user_id: str = "default", force_new: bool = False) -> List[Dict[str, Any]]:
    """Generate a 30-day content calendar using Gemini AI with real-time trends"""
    logger.info(f"Generating content calendar for user: {user_id}, niche: {request.niche}")
    
    # Generate a unique hash for this request
    request_hash = generate_request_hash(request)
    cache_key = f"{user_id}_{request_hash}"
    
    # Check if we already have a calendar for this user and request
    if not force_new and cache_key in user_calendars:
        calendar_data = user_calendars[cache_key]
        last_generated = datetime.fromisoformat(calendar_data["last_generated"])
        
        # If we generated a calendar recently, return it (cache for 1 hour)
        if (datetime.now() - last_generated).total_seconds() < 3600:
            logger.info(f"Returning cached calendar for user: {user_id}")
            return calendar_data["calendar"]
    
    # Get real-time trends
    logger.info("Fetching real-time trends for content generation")
    search_trends = await get_search_trends(request.niche)
    youtube_trends = await get_youtube_trends(request.niche)
    
    # Combine all trends
    all_trends = list(set(search_trends + youtube_trends))
    logger.info(f"Combined {len(all_trends)} trends")
    logger.debug(safe_log_message(f"Combined trends: {all_trends}"))
    
    # Create a JSON output parser
    parser = JsonOutputParser()
    
    # Get tomorrow's date for starting the calendar
    tomorrow = datetime.now().date() + timedelta(days=1)
    
    # Create the prompt template with real-time trends - FIXED THE PROMPT TEMPLATE
    prompt = PromptTemplate(
        template="""
        As an expert content strategist, create a diverse 30-day content calendar for a content creator.
        
        CONTENT TYPE: {content_type}
        PLATFORMS: {platforms}
        NICHE/TOPIC: {niche}
        TARGET AUDIENCE: {audience}
        GOALS: {goals}
        
        CURRENT TRENDS IN THIS NICHE: {trends}
        
        Generate a diverse content calendar with:
        1. Unique, engaging titles for each day that incorporate current trends
        2. Brief descriptions (1-2 sentences) that explain the content
        3. Appropriate platform for each content piece (rotate through the selected platforms)
        4. Relevant tags/hashtags (3-5 per piece) that include trending keywords
        5. Spread content evenly across 30 days starting from {start_date}
        
        Ensure each day has completely different content. Vary the:
        - Topics within the niche
        - Content formats (tutorials, reviews, interviews, etc.)
        - Angles (beginner vs advanced, problem/solution, etc.)
        
        Incorporate current trends from the provided list to make content more relevant.
        
        Return ONLY a valid JSON array without any markdown or extra text. Each item should have:
        - title: string (unique and engaging)
        - description: string
        - platform: string (from: {platforms})
        - publish_date: string (YYYY-MM-DD, starting from {start_date})
        - tags: array of strings (3-5 relevant tags)
        
        Example format:
        [
          {{
            "title": "Beginner's Guide to {{$niche}}: Getting Started with Trending Techniques",
            "description": "Learn the fundamentals of {{$niche}} and how to get started as a complete beginner.",
            "platform": "YouTube",
            "publish_date": "{start_date}",
            "tags": ["beginner", "tutorial", "guide", "{{$niche}}", "trending"]
          }},
          {{
            "title": "Advanced Techniques for Mastering {{$niche}}",
            "description": "Discover professional techniques that will take your {{$niche}} skills to the next level.",
            "platform": "Blog",
            "publish_date": "{next_date}",
            "tags": ["advanced", "techniques", "master", "{{$niche}}", "professional"]
          }}
        ]
        """,
        input_variables=["content_type", "platforms", "niche", "audience", "goals", "trends"],
        partial_variables={
            "start_date": tomorrow.strftime("%Y-%m-%d"),
            "next_date": (tomorrow + timedelta(days=1)).strftime("%Y-%m-%d")
        }
    )
    
    # Create the chain
    chain = prompt | llm | parser
    
    # Store model invocation details for debugging
    model_debug = {
        "prompt": prompt.template,
        "input_variables": {
            "content_type": request.content_type,
            "platforms": ", ".join(request.platforms),
            "niche": request.niche,
            "audience": request.audience,
            "goals": request.goals,
            "trends": ", ".join(all_trends[:5])  # Only use top 5 trends to avoid token limits
        },
        "status": "started"
    }
    
    try:
        # Log the model input
        logger.info("Invoking model with input (truncated for logs)")
        logger.debug(safe_log_message(f"Model input: {model_debug['input_variables']}"))
        
        # Invoke the chain with trends
        result = chain.invoke({
            "content_type": request.content_type,
            "platforms": ", ".join(request.platforms),
            "niche": request.niche,
            "audience": request.audience,
            "goals": request.goals,
            "trends": ", ".join(all_trends[:5])  # Only use top 5 trends to avoid token limits
        })
        
        # Update debug info with successful response
        model_debug["status"] = "success"
        model_debug["response"] = result
        model_debug["trends_used"] = all_trends[:5]
        debug_info["model_invocation"] = model_debug
        
        logger.info(f"Model invocation successful, generated {len(result)} items")
        
        # Validate and ensure we have 30 unique items
        if len(result) < 30:
            # If we don't have enough items, supplement with fallback
            fallback_items = generate_fallback_calendar(request)
            # Use AI-generated items first, then supplement with fallback
            result.extend(fallback_items[len(result):])
            logger.warning(f"Model generated only {len(result)} items, added {30 - len(result)} fallback items")
        elif len(result) > 30:
            # If we have too many, trim to 30
            result = result[:30]
            logger.info(f"Model generated {len(result)} items, trimmed to 30")
        
        # Store the calendar for future use
        user_calendars[cache_key] = {
            "calendar": result,
            "last_generated": datetime.now().isoformat(),
            "request": request.dict()
        }
        
        logger.info(f"Content calendar generated successfully for user: {user_id}")
        return result
        
    except OutputParserException as e:
        logger.error(f"Output parsing error: {e}")
        # Update debug info with error
        model_debug["status"] = "output_parse_error"
        model_debug["error"] = str(e)
        debug_info["model_invocation"] = model_debug
        
        # Fallback to a default calendar if parsing fails
        logger.warning("Using fallback calendar due to parsing error")
        return generate_fallback_calendar(request)
    except Exception as e:
        logger.error(f"Error generating content calendar: {e}")
        # Update debug info with error
        model_debug["status"] = "error"
        model_debug["error"] = str(e)
        debug_info["model_invocation"] = model_debug
        
        # Fallback to a default calendar if AI fails
        logger.warning("Using fallback calendar due to generation error")
        return generate_fallback_calendar(request)

def generate_fallback_calendar(request: ContentRequest) -> List[Dict[str, Any]]:
    """Generate a fallback content calendar with diverse content"""
    logger.info("Generating fallback content calendar")
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
    
    logger.info("Fallback calendar generated successfully")
    return calendar

async def detect_trends(niche: str) -> List[str]:
    """Detect current trends in the given niche using multiple sources"""
    logger.info(f"Detecting trends for niche: {niche}")
    
    # Get trends from multiple sources
    search_trends = await get_search_trends(niche)
    youtube_trends = await get_youtube_trends(niche)
    
    # Combine and deduplicate trends
    all_trends = list(set(search_trends + youtube_trends))
    
    # If we have enough trends, return them
    if len(all_trends) >= 3:
        logger.info(f"Using API trends: {len(all_trends)} trends found")
        logger.debug(safe_log_message(f"API trends: {all_trends[:5]}"))
        return all_trends[:5]  # Return top 5 trends
    
    # Fallback to AI if we don't have enough trends from APIs
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
    
    # Store trend detection model invocation details for debugging
    trend_debug = {
        "prompt": prompt.template,
        "input_variables": {"niche": niche},
        "status": "started"
    }
    
    try:
        logger.info("Not enough API trends found, using AI for trend detection")
        result = chain.invoke({"niche": niche})
        
        # Update debug info with successful response
        trend_debug["status"] = "success"
        trend_debug["response"] = result
        debug_info["trend_detection"] = trend_debug
        
        logger.info(f"AI trend detection successful: {len(result)} trends found")
        logger.debug(safe_log_message(f"AI trends: {result}"))
        return result
    except Exception as e:
        logger.error(f"Error detecting trends: {e}")
        
        # Update debug info with error
        trend_debug["status"] = "error"
        trend_debug["error"] = str(e)
        debug_info["trend_detection"] = trend_debug
        
        logger.warning("Using fallback trends")
        return [f"Trend in {niche}", "Industry News", "How-to Guides", "Product Reviews", "Expert Opinions"]

# FastAPI routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    logger.info("Serving index page")
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
    force_new: bool = Form(False)
):
    logger.info(f"Received calendar generation request: {niche}, {content_type}, {platforms}")
    try:
        content_request = ContentRequest(
            content_type=content_type,
            platforms=platforms,
            niche=niche,
            audience=audience,
            goals=goals
        )
        
        # Generate content calendar with real-time trends
        user_id = request.client.host if request.client else "default"
        calendar = await generate_content_calendar(content_request, user_id, force_new)
        
        # Detect trends from multiple sources
        trends = await detect_trends(niche)
        
        logger.info(f"Calendar generation completed successfully for {niche}")
        
        return JSONResponse({
            "success": True,
            "calendar": calendar,
            "trends": trends,
            "generated_date": datetime.now().isoformat(),
            "debug": debug_info  # Add debug info to the response
        })
        
    except Exception as e:
        logger.error(f"Error generating content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating content: {str(e)}")

@app.get("/health")
async def health_check():
    logger.debug("Health check endpoint called")
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/debug-trends")
async def debug_trends(niche: str = "technology"):
    """Debug endpoint to check if search trends are working"""
    logger.info(f"Debug trends endpoint called for niche: {niche}")
    # Clear previous debug info
    global debug_info
    debug_info = {}
    
    # Test search trends
    search_results = await get_search_trends(niche)
    
    # Test YouTube trends
    youtube_results = await get_youtube_trends(niche)
    
    # Return detailed debug information
    return {
        "niche": niche,
        "search_trends": search_results,
        "youtube_trends": youtube_results,
        "debug_info": debug_info,
        "api_keys": {
            "serpapi_found": bool(serpapi_key),
            "youtube_found": bool(youtube_key)
        }
    }

@app.get("/debug-model")
async def debug_model(
    niche: str = "technology",
    content_type: str = "Educational",
    platforms: str = "YouTube,Blog",
    audience: str = "beginners",
    goals: str = "education and engagement"
):
    """Debug endpoint to test the model with API trends"""
    logger.info(f"Debug model endpoint called: {niche}, {content_type}, {platforms}")
    # Clear previous debug info
    global debug_info
    debug_info = {}
    
    # Create a test request
    test_request = ContentRequest(
        content_type=content_type,
        platforms=platforms.split(","),
        niche=niche,
        audience=audience,
        goals=goals
    )
    
    # Generate content calendar
    calendar = await generate_content_calendar(test_request, "debug_user", True)
    
    # Return detailed debug information
    return {
        "request": test_request.dict(),
        "calendar": calendar,
        "debug_info": debug_info
    }

@app.get("/refresh-calendar")
async def refresh_calendar(request: Request):
    """Force refresh of the content calendar for a user"""
    user_id = request.client.host if request.client else "default"
    logger.info(f"Refreshing calendar for user: {user_id}")
    
    # Remove all cached calendars for this user
    keys_to_remove = [key for key in user_calendars.keys() if key.startswith(user_id)]
    for key in keys_to_remove:
        del user_calendars[key]
        
    logger.info(f"Removed {len(keys_to_remove)} cached calendars for user: {user_id}")
    return {"status": "refreshed", "user_id": user_id, "removed_entries": len(keys_to_remove)}

@app.on_event("startup")
async def startup_event():
    logger.info("Content Platform API starting up")
    logger.info(f"Available content types: {CONTENT_TYPES}")
    logger.info(f"Available platforms: {PLATFORMS}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Content Platform API shutting down")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True)
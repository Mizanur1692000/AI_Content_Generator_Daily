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
from fastapi import FastAPI, Request, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
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

# Default content types and platforms
DEFAULT_CONTENT_TYPES = ["Educational", "Entertainment", "How-to Guide", "Review", "News", "Opinion", "Listicle", "Custom"]
DEFAULT_PLATFORMS = ["YouTube", "Blog", "Instagram", "Twitter", "TikTok", "LinkedIn"]

# Store dynamic content types, angles, and formats with caching
dynamic_content_cache = {
    "types": {},
    "angles": {},
    "formats": {},
    "trends": {}
}
CACHE_DURATION = 3600  # 1 hour

# Store debug information for troubleshooting
debug_info = {}

# Pydantic models
class ContentRequest(BaseModel):
    content_type: str
    platforms: List[str]
    niche: str
    audience: str
    goals: str
    custom_content_type: Optional[str] = None

class ContentIdea(BaseModel):
    title: str
    description: str
    platform: str
    publish_date: str
    tags: List[str]

# Store user content calendars with request hashing
user_calendars = {}

# Store endpoint usage statistics
endpoint_stats = {
    "total_requests": 0,
    "endpoints": {},
    "model_invocations": 0,
    "api_calls": {
        "serpapi": 0,
        "youtube": 0,
        "gemini": 0
    }
}

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

def track_endpoint_usage(endpoint_name):
    """Track endpoint usage statistics"""
    endpoint_stats["total_requests"] += 1
    
    if endpoint_name not in endpoint_stats["endpoints"]:
        endpoint_stats["endpoints"][endpoint_name] = {
            "count": 0,
            "last_called": None,
            "errors": 0
        }
    
    endpoint_stats["endpoints"][endpoint_name]["count"] += 1
    endpoint_stats["endpoints"][endpoint_name]["last_called"] = datetime.now().isoformat()

def track_model_invocation(model_name):
    """Track model invocation statistics"""
    endpoint_stats["model_invocations"] += 1
    endpoint_stats["api_calls"][model_name] += 1

async def validate_api_keys():
    """Validate all API keys and log their status"""
    logger.info("Validating API keys...")
    
    # Test Gemini API
    gemini_status = "❌ Not working"
    if api_key:
        try:
            model = genai.GenerativeModel('gemini-2.5-pro')
            response = model.generate_content("Test connection")
            if response.text:
                gemini_status = "✅ Working"
            else:
                gemini_status = "❌ No response text"
        except Exception as e:
            gemini_status = f"❌ Error: {str(e)}"
    else:
        gemini_status = "❌ No API key found"
    
    # Test SerpAPI
    serpapi_status = "❌ Not working"
    if serpapi_key:
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    'engine': 'google',
                    'q': 'test',
                    'api_key': serpapi_key
                }
                async with session.get('https://serpapi.com/search', params=params) as response:
                    if response.status == 200:
                        serpapi_status = "✅ Working"
                    else:
                        serpapi_status = f"❌ HTTP {response.status}"
        except Exception as e:
            serpapi_status = f"❌ Error: {str(e)}"
    else:
        serpapi_status = "❌ No API key found"
    
    # Test YouTube API
    youtube_status = "❌ Not working"
    if youtube_key:
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    'part': 'snippet',
                    'q': 'test',
                    'type': 'video',
                    'maxResults': 1,
                    'key': youtube_key
                }
                async with session.get('https://www.googleapis.com/youtube/v3/search', params=params) as response:
                    if response.status == 200:
                        youtube_status = "✅ Working"
                    else:
                        youtube_status = f"❌ HTTP {response.status}"
        except Exception as e:
            youtube_status = f"❌ Error: {str(e)}"
    else:
        youtube_status = "❌ No API key found"
    
    # Log the results
    logger.info(f"Gemini API Status: {gemini_status}")
    logger.info(f"SerpAPI Status: {serpapi_status}")
    logger.info(f"YouTube API Status: {youtube_status}")
    
    return {
        "gemini": gemini_status,
        "serpapi": serpapi_status,
        "youtube": youtube_status
    }

def clean_output(text: str) -> str:
    """Clean the model output by removing markdown code blocks if present."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text

async def get_search_trends(niche: str) -> List[str]:
    global debug_info
    """Get current search trends using SerpAPI"""
    logger.info(f"Fetching search trends for niche: {niche}")
    track_model_invocation("serpapi")
    
    # Check cache first
    cache_key = f"search_trends_{niche.lower()}"
    if cache_key in dynamic_content_cache["trends"]:
        cached_data = dynamic_content_cache["trends"][cache_key]
        if (datetime.now() - cached_data["timestamp"]).total_seconds() < CACHE_DURATION:
            logger.info(f"Using cached search trends for: {niche}")
            return cached_data["data"]
    
    trends = []
    debug_data = {"api": "serpapi", "niche": niche, "status": "started", "key_found": bool(serpapi_key)}
    
    if not serpapi_key:
        logger.warning("SERPAPI_KEY not found, using fallback trends")
        debug_data["status"] = "no_api_key"
        debug_data["fallback_trends"] = [f"Trend in {niche}", "Industry News", "How-to Guides", "Product Reviews", "Expert Opinions"]
        # Store debug info
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
            
            async with session.get('https://serpapi.com/search', params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
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
                    async with session.get('https://serpapi.com/search', params=params, timeout=aiohttp.ClientTimeout(total=10)) as rel_response:
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
                    # Store debug info
                    debug_info["search_trends"] = debug_data
                    
                    # Cache the results
                    dynamic_content_cache["trends"][cache_key] = {
                        "data": list(set(trends))[:10],  # Return unique trends, max 10
                        "timestamp": datetime.now()
                    }
                    
                    logger.info(f"Found {len(trends)} search trends for {niche}")
                    logger.debug(safe_log_message(f"Search trends: {trends}"))
                    return dynamic_content_cache["trends"][cache_key]["data"]
                    
    except Exception as e:
        debug_data["status"] = "error"
        debug_data["error"] = str(e)
        # Store debug info
        debug_info["search_trends"] = debug_data
        logger.error(f"Error fetching search trends: {e}")
        return [f"Trend in {niche}", "Industry News", "How-to Guides", "Product Reviews", "Expert Opinions"]

async def get_youtube_trends(niche: str) -> List[str]:
    global debug_info
    """Get YouTube trending videos related to the niche"""
    logger.info(f"Fetching YouTube trends for niche: {niche}")
    track_model_invocation("youtube")
    
    # Check cache first
    cache_key = f"youtube_trends_{niche.lower()}"
    if cache_key in dynamic_content_cache["trends"]:
        cached_data = dynamic_content_cache["trends"][cache_key]
        if (datetime.now() - cached_data["timestamp"]).total_seconds() < CACHE_DURATION:
            logger.info(f"Using cached YouTube trends for: {niche}")
            return cached_data["data"]
    
    trends = []
    debug_data = {"api": "youtube", "niche": niche, "status": "started", "key_found": bool(youtube_key)}
    
    if not youtube_key:
        logger.warning("YOUTUBE_API_KEY not found, using fallback trends")
        debug_data["status"] = "no_api_key"
        debug_data["fallback_trends"] = [f"{niche} videos", f"{niche} tutorials", f"{niche} reviews"]
        # Store debug info
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
            
            async with session.get('https://www.googleapis.com/youtube/v3/search', params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
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
                    # Store debug info
                    debug_info["youtube_trends"] = debug_data
                    
                    # Cache the results
                    dynamic_content_cache["trends"][cache_key] = {
                        "data": trends[:5],  # Return max 5 trends
                        "timestamp": datetime.now()
                    }
                    
                    logger.info(f"Found {len(trends)} YouTube trends for {niche}")
                    logger.debug(safe_log_message(f"YouTube trends: {trends}"))
                    return dynamic_content_cache["trends"][cache_key]["data"]
                    
    except Exception as e:
        debug_data["status"] = "error"
        debug_data["error"] = str(e)
        # Store debug info
        debug_info["youtube_trends"] = debug_data
        logger.error(f"Error fetching YouTube trends: {e}")
        return [f"{niche} videos", f"{niche} tutorials", f"{niche} reviews"]

async def get_dynamic_content_types(niche: str, audience: str) -> List[str]:
    """Get dynamic content types based on niche and audience using SerpAPI"""
    logger.info(f"Getting dynamic content types for niche: {niche}, audience: {audience}")
    
    # Check cache first
    cache_key = f"content_types_{niche.lower()}_{audience.lower()}"
    if cache_key in dynamic_content_cache["types"]:
        cached_data = dynamic_content_cache["types"][cache_key]
        if (datetime.now() - cached_data["timestamp"]).total_seconds() < CACHE_DURATION:
            logger.info(f"Using cached content types for: {niche}")
            return cached_data["data"]
    
    content_types = DEFAULT_CONTENT_TYPES.copy()
    
    if not serpapi_key:
        logger.warning("SERPAPI_KEY not found, using default content types")
        return content_types
    
    try:
        # Search for popular content types in this niche
        async with aiohttp.ClientSession() as session:
            params = {
                'engine': 'google',
                'q': f'{niche} content types',
                'api_key': serpapi_key
            }
            
            async with session.get('https://serpapi.com/search', params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract content types from search results
                    if 'organic_results' in data:
                        for result in data['organic_results']:
                            if 'title' in result:
                                title = result['title']
                                # Extract potential content types from titles
                                if 'how to' in title.lower() or 'tutorial' in title.lower():
                                    if 'How-to Guide' not in content_types:
                                        content_types.append('How-to Guide')
                                elif 'review' in title.lower():
                                    if 'Review' not in content_types:
                                        content_types.append('Review')
                                elif 'news' in title.lower():
                                    if 'News' not in content_types:
                                        content_types.append('News')
                                elif 'list' in title.lower() or 'listicle' in title.lower():
                                    if 'Listicle' not in content_types:
                                        content_types.append('Listicle')
                    
                    # Cache the results
                    dynamic_content_cache["types"][cache_key] = {
                        "data": content_types,
                        "timestamp": datetime.now()
                    }
                    
                    logger.info(f"Found {len(content_types)} content types for {niche}")
                    return content_types
                    
    except Exception as e:
        logger.error(f"Error fetching dynamic content types: {e}")
        return content_types
    
    return content_types

async def get_dynamic_content_angles(niche: str, content_type: str) -> List[str]:
    """Get dynamic content angles based on niche and content type using SerpAPI"""
    logger.info(f"Getting dynamic content angles for niche: {niche}, type: {content_type}")
    
    # Check cache first
    cache_key = f"content_angles_{niche.lower()}_{content_type.lower()}"
    if cache_key in dynamic_content_cache["angles"]:
        cached_data = dynamic_content_cache["angles"][cache_key]
        if (datetime.now() - cached_data["timestamp"]).total_seconds() < CACHE_DURATION:
            logger.info(f"Using cached content angles for: {niche}")
            return cached_data["data"]
    
    content_angles = []
    
    if not serpapi_key:
        logger.warning("SERPAPI_KEY not found, using AI-generated content angles")
        return await generate_ai_content_angles(niche, content_type)
    
    try:
        # Search for popular content angles in this niche and content type
        async with aiohttp.ClientSession() as session:
            params = {
                'engine': 'google',
                'q': f'{niche} {content_type} content angles',
                'api_key': serpapi_key
            }
            
            async with session.get('https://serpapi.com/search', params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract content angles from search results
                    if 'organic_results' in data:
                        for result in data['organic_results']:
                            if 'title' in result:
                                title = result['title']
                                # Extract potential content angles from titles
                                if 'beginner' in title.lower() and 'guide' in title.lower():
                                    if "Beginner's Guide to" not in content_angles:
                                        content_angles.append("Beginner's Guide to")
                                elif 'advanced' in title.lower() and 'techniques' in title.lower():
                                    if "Advanced Techniques for" not in content_angles:
                                        content_angles.append("Advanced Techniques for")
                                elif 'ultimate guide' in title.lower():
                                    if "The Ultimate Guide to" not in content_angles:
                                        content_angles.append("The Ultimate Guide to")
                                elif 'mistakes' in title.lower() and 'avoid' in title.lower():
                                    if "Mistakes to Avoid with" not in content_angles:
                                        content_angles.append("Mistakes to Avoid with")
                                elif 'surprising facts' in title.lower():
                                    if "5 Surprising Facts About" not in content_angles:
                                        content_angles.append("5 Surprising Facts About")
                                elif 'expert interview' in title.lower():
                                    if "Interview with an Expert on" not in content_angles:
                                        content_angles.append("Interview with an Expert on")
                    
                    # If we don't get enough angles, supplement with AI-generated ones
                    if len(content_angles) < 5:
                        ai_angles = await generate_ai_content_angles(niche, content_type)
                        content_angles.extend(ai_angles)
                    
                    # Cache the results
                    dynamic_content_cache["angles"][cache_key] = {
                        "data": content_angles,
                        "timestamp": datetime.now()
                    }
                    
                    logger.info(f"Found {len(content_angles)} content angles for {niche}")
                    return content_angles
                    
    except Exception as e:
        logger.error(f"Error fetching dynamic content angles: {e}")
        return await generate_ai_content_angles(niche, content_type)
    
    return content_angles

async def get_dynamic_content_formats(niche: str, platform: str) -> List[str]:
    """Get dynamic content formats based on niche and platform using SerpAPI"""
    logger.info(f"Getting dynamic content formats for niche: {niche}, platform: {platform}")
    
    # Check cache first
    cache_key = f"content_formats_{niche.lower()}_{platform.lower()}"
    if cache_key in dynamic_content_cache["formats"]:
        cached_data = dynamic_content_cache["formats"][cache_key]
        if (datetime.now() - cached_data["timestamp"]).total_seconds() < CACHE_DURATION:
            logger.info(f"Using cached content formats for: {niche}")
            return cached_data["data"]
    
    content_formats = []
    
    if not serpapi_key:
        logger.warning("SERPAPI_KEY not found, using AI-generated content formats")
        return await generate_ai_content_formats(niche, platform)
    
    try:
        # Search for popular content formats in this niche and platform
        async with aiohttp.ClientSession() as session:
            params = {
                'engine': 'google',
                'q': f'{niche} {platform} content formats',
                'api_key': serpapi_key
            }
            
            async with session.get('https://serpapi.com/search', params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract content formats from search results
                    if 'organic_results' in data:
                        for result in data['organic_results']:
                            if 'title' in result:
                                title = result['title']
                                # Extract potential content formats from titles
                                if 'tutorial' in title.lower():
                                    if "Step-by-Step Tutorial" not in content_formats:
                                        content_formats.append("Step-by-Step Tutorial")
                                elif 'q&a' in title.lower() or 'q and a' in title.lower():
                                    if "Q&A Session" not in content_formats:
                                        content_formats.append("Q&A Session")
                                elif 'live' in title.lower() and 'demo' in title.lower():
                                    if "Live Demonstration" not in content_formats:
                                        content_formats.append("Live Demonstration")
                                elif 'comparison' in title.lower() and 'review' in title.lower():
                                    if "Comparison Review" not in content_formats:
                                        content_formats.append("Comparison Review")
                                elif 'showcase' in title.lower():
                                    if "Product Showcase" not in content_formats:
                                        content_formats.append("Product Showcase")
                                elif 'analysis' in title.lower():
                                    if "Industry Analysis" not in content_formats:
                                        content_formats.append("Industry Analysis")
                    
                    # If we don't get enough formats, supplement with AI-generated ones
                    if len(content_formats) < 5:
                        ai_formats = await generate_ai_content_formats(niche, platform)
                        content_formats.extend(ai_formats)
                    
                    # Cache the results
                    dynamic_content_cache["formats"][cache_key] = {
                        "data": content_formats,
                        "timestamp": datetime.now()
                    }
                    
                    logger.info(f"Found {len(content_formats)} content formats for {niche}")
                    return content_formats
                    
    except Exception as e:
        logger.error(f"Error fetching dynamic content formats: {e}")
        return await generate_ai_content_formats(niche, platform)
    
    return content_formats

async def generate_ai_content_angles(niche: str, content_type: str) -> List[str]:
    """Generate content angles using AI when API is not available"""
    logger.info(f"Generating AI content angles for niche: {niche}, type: {content_type}")
    
    prompt = PromptTemplate(
        template="""Generate 10 creative content angles for {content_type} content in the {niche} niche.

Return ONLY a JSON array of angle names without any additional text or markdown.

Example: ["Beginner's Guide to {niche}", "Advanced Techniques for {niche}", "The Ultimate Guide to {niche}"]""",
        input_variables=["niche", "content_type"]
    )
    
    # Create the chain
    chain = prompt | llm | RunnableLambda(lambda x: clean_output(x.content)) | JsonOutputParser()
    
    try:
        track_model_invocation("gemini")
        result = chain.invoke({"niche": niche, "content_type": content_type})
        logger.info(f"AI generated {len(result)} content angles for {niche}")
        return result
    except Exception as e:
        logger.error(f"Error generating AI content angles: {e}")
        # Fallback to basic angles
        return [
            f"Beginner's Guide to {niche}",
            f"Advanced Techniques for {niche}",
            f"The Ultimate Guide to {niche}",
            f"5 Surprising Facts About {niche}",
            f"Mistakes to Avoid with {niche}",
            f"Interview with an Expert on {niche}",
            f"Case Study: {niche}",
            f"Behind the Scenes of {niche}",
            f"Future of {niche}",
            f"History of {niche}"
        ]

async def generate_ai_content_formats(niche: str, platform: str) -> List[str]:
    """Generate content formats using AI when API is not available"""
    logger.info(f"Generating AI content formats for niche: {niche}, platform: {platform}")
    
    prompt = PromptTemplate(
        template="""Generate 10 creative content formats for {platform} content in the {niche} niche.

Return ONLY a JSON array of format names without any additional text or markdown.

Example: ["Step-by-Step Tutorial", "Q&A Session", "Live Demonstration", "Comparison Review"]""",
        input_variables=["niche", "platform"]
    )
    
    # Create the chain
    chain = prompt | llm | RunnableLambda(lambda x: clean_output(x.content)) | JsonOutputParser()
    
    try:
        track_model_invocation("gemini")
        result = chain.invoke({"niche": niche, "platform": platform})
        logger.info(f"AI generated {len(result)} content formats for {niche}")
        return result
    except Exception as e:
        logger.error(f"Error generating AI content formats: {e}")
        # Fallback to basic formats
        return [
            "Step-by-Step Tutorial",
            "Q&A Session",
            "Live Demonstration",
            "Comparison Review",
            "Product Showcase",
            "Industry Analysis",
            "Personal Story",
            "Data-Driven Insights",
            "Collaboration with Experts",
            "Challenge/Accepted"
        ]

async def generate_content_calendar(request: ContentRequest, user_id: str = "default", force_new: bool = False) -> List[Dict[str, Any]]:
    global debug_info
    """Generate a 30-day content calendar using Gemini AI with real-time trends"""
    logger.info(f"Generating content calendar for user: {user_id}, niche: {request.niche}")
    
    # Initialize debug info for this request
    debug_info = {}
    
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
    
    # Get real-time trends and dynamic content options in parallel
    logger.info("Fetching real-time trends and dynamic content options")
    
    # Run all API calls in parallel
    search_trends_task = get_search_trends(request.niche)
    youtube_trends_task = get_youtube_trends(request.niche)
    content_types_task = get_dynamic_content_types(request.niche, request.audience)
    content_angles_task = get_dynamic_content_angles(request.niche, request.content_type)
    
    # For formats, use the first platform or a default
    target_platform = request.platforms[0] if request.platforms else "YouTube"
    content_formats_task = get_dynamic_content_formats(request.niche, target_platform)
    
    # Wait for all tasks to complete
    search_trends, youtube_trends, content_types, content_angles, content_formats = await asyncio.gather(
        search_trends_task, youtube_trends_task, content_types_task, content_angles_task, content_formats_task
    )
    
    # Combine all trends
    all_trends = list(set(search_trends + youtube_trends))
    logger.info(f"Combined {len(all_trends)} trends")
    logger.debug(safe_log_message(f"Combined trends: {all_trends}"))
    
    # Add custom content type if provided
    if request.custom_content_type and request.custom_content_type not in content_types:
        content_types.append(request.custom_content_type)
    
    # Create a JSON output parser
    parser = JsonOutputParser()
    
    # Get tomorrow's date for starting the calendar
    tomorrow = datetime.now().date() + timedelta(days=1)
    
    # Create the prompt template with real-time trends
    prompt = PromptTemplate(
        template="""As an expert content strategist, create a diverse 30-day content calendar for a content creator.

CONTENT TYPE: {content_type}
PLATFORMS: {platforms}
NICHE/TOPIC: {niche}
TARGET AUDIENCE: {audience}
GOALS: {goals}
CURRENT TRENDS IN THIS NICHE: {trends}
CONTENT ANGLES TO USE: {angles}
CONTENT FORMATS TO USE: {formats}

Generate a diverse content calendar with:
1. Unique, engaging titles for each day that incorporate current trends
2. Brief descriptions (1-2 sentences) that explain the content
3. Appropriate platform for each content piece (rotate through the selected platforms)
4. Relevant tags/hashtags (3-5 per piece) that include trending keywords
5. Spread content evenly across 30 days starting from {start_date}

Ensure each day has completely different content. Vary the:
- Topics within the niche
- Content formats (use the provided formats)
- Angles (use the provided angles)

Incorporate current trends from the provided list to make content more relevant.

Return ONLY a valid JSON array without any markdown, code blocks, or extra text. Each item should have:
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
        input_variables=["content_type", "platforms", "niche", "audience", "goals", "trends", "angles", "formats"],
        partial_variables={
            "start_date": tomorrow.strftime("%Y-%m-%d"),
            "next_date": (tomorrow + timedelta(days=1)).strftime("%Y-%m-%d")
        }
    )
    
    # Create the chain with cleaning step
    chain = prompt | llm | RunnableLambda(lambda x: clean_output(x.content)) | parser
    
    # Store model invocation details for debugging
    model_debug = {
        "prompt": prompt.template,
        "input_variables": {
            "content_type": request.content_type,
            "platforms": ", ".join(request.platforms),
            "niche": request.niche,
            "audience": request.audience,
            "goals": request.goals,
            "trends": ", ".join(all_trends[:5]),  # Only use top 5 trends to avoid token limits
            "angles": ", ".join(content_angles[:10]),  # Use top 10 angles
            "formats": ", ".join(content_formats[:10])  # Use top 10 formats
        },
        "status": "started"
    }
    
    try:
        # Log the model input
        logger.info("Invoking model with input (truncated for logs)")
        logger.debug(safe_log_message(f"Model input: {model_debug['input_variables']}"))
        
        # Track model invocation
        track_model_invocation("gemini")
        
        # Invoke the chain with trends
        result = chain.invoke({
            "content_type": request.content_type,
            "platforms": ", ".join(request.platforms),
            "niche": request.niche,
            "audience": request.audience,
            "goals": request.goals,
            "trends": ", ".join(all_trends[:5]),  # Only use top 5 trends to avoid token limits
            "angles": ", ".join(content_angles[:10]),  # Use top 10 angles
            "formats": ", ".join(content_formats[:10])  # Use top 10 formats
        })
        
        # Update debug info with successful response
        model_debug["status"] = "success"
        model_debug["response"] = result
        model_debug["trends_used"] = all_trends[:5]
        model_debug["angles_used"] = content_angles[:10]
        model_debug["formats_used"] = content_formats[:10]
        debug_info["model_invocation"] = model_debug
        
        logger.info(f"Model invocation successful, generated {len(result)} items")
        
        # Validate and ensure we have 30 unique items
        if len(result) < 30:
            # If we don't have enough items, supplement with fallback
            fallback_items = generate_fallback_calendar(request, content_angles, content_formats)
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
        return generate_fallback_calendar(request, content_angles, content_formats)
        
    except Exception as e:
        logger.error(f"Error generating content calendar: {e}")
        # Update debug info with error
        model_debug["status"] = "error"
        model_debug["error"] = str(e)
        debug_info["model_invocation"] = model_debug
        
        # Fallback to a default calendar if AI fails
        logger.warning("Using fallback calendar due to generation error")
        return generate_fallback_calendar(request, content_angles, content_formats)

def generate_fallback_calendar(request: ContentRequest, angles: List[str], formats: List[str]) -> List[Dict[str, Any]]:
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
        angle = random.choice(angles) if angles else f"Content about {request.niche}"
        content_format = random.choice(formats) if formats else "Content Format"
        
        # Create unique content for each day
        title = f"{angle}: {content_format}"
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
    global debug_info
    """Detect current trends in the given niche using multiple sources"""
    logger.info(f"Detecting trends for niche: {niche}")
    
    # Initialize debug info for this request
    debug_info = {}
    
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
        template="""As an AI trend detector, identify the top 5 current trends in the following niche: {niche}

Return ONLY a JSON array of trend names without any additional text or markdown.

Example: ["Trend 1", "Trend 2", "Trend 3"]""",
        input_variables=["niche"]
    )
    
    # Create the chain
    chain = prompt | llm | RunnableLambda(lambda x: clean_output(x.content)) | JsonOutputParser()
    
    # Store trend detection model invocation details for debugging
    trend_debug = {
        "prompt": prompt.template,
        "input_variables": {"niche": niche},
        "status": "started"
    }
    
    try:
        logger.info("Not enough API trends found, using AI for trend detection")
        track_model_invocation("gemini")
        
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
    track_endpoint_usage("root")
    logger.info("Serving index page")
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "content_types": DEFAULT_CONTENT_TYPES,
            "platforms": DEFAULT_PLATFORMS
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
    custom_content_type: Optional[str] = Form(None),
    force_new: bool = Form(False)
):
    track_endpoint_usage("generate-calendar")
    logger.info(f"Received calendar generation request: {niche}, {content_type}, {platforms}")
    
    try:
        content_request = ContentRequest(
            content_type=content_type,
            platforms=platforms,
            niche=niche,
            audience=audience,
            goals=goals,
            custom_content_type=custom_content_type
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
            "generated_date": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error generating content: {str(e)}")
        # Track error in endpoint stats
        if "generate-calendar" in endpoint_stats["endpoints"]:
            endpoint_stats["endpoints"]["generate-calendar"]["errors"] += 1
        raise HTTPException(status_code=500, detail=f"Error generating content: {str(e)}")

@app.get("/health")
async def health_check():
    track_endpoint_usage("health")
    logger.debug("Health check endpoint called")
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/debug-trends")
async def debug_trends(niche: str = "technology"):
    global debug_info
    """Debug endpoint to check if search trends are working"""
    track_endpoint_usage("debug-trends")
    logger.info(f"Debug trends endpoint called for niche: {niche}")
    
    # Clear previous debug info
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
    global debug_info
    """Debug endpoint to test the model with API trends"""
    track_endpoint_usage("debug-model")
    logger.info(f"Debug model endpoint called: {niche}, {content_type}, {platforms}")
    
    # Clear previous debug info
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
    track_endpoint_usage("refresh-calendar")
    user_id = request.client.host if request.client else "default"
    logger.info(f"Refreshing calendar for user: {user_id}")
    
    # Remove all cached calendars for this user
    keys_to_remove = [key for key in user_calendars.keys() if key.startswith(user_id)]
    for key in keys_to_remove:
        del user_calendars[key]
    
    logger.info(f"Removed {len(keys_to_remove)} cached calendars for user: {user_id}")
    return {"status": "refreshed", "user_id": user_id, "removed_entries": len(keys_to_remove)}

@app.get("/api-status")
async def api_status():
    """Check status of all APIs and endpoints"""
    track_endpoint_usage("api-status")
    logger.info("API status check requested")
    
    # Test Gemini API
    gemini_status = "unknown"
    try:
        # Simple test to check if Gemini API is working
        model = genai.GenerativeModel('gemini-2.5-pro')
        response = model.generate_content("Hello")
        gemini_status = "working" if response.text else "error"
    except Exception as e:
        gemini_status = f"error: {str(e)}"
    
    # Test SerpAPI
    serpapi_status = "unknown"
    try:
        if serpapi_key:
            async with aiohttp.ClientSession() as session:
                params = {
                    'engine': 'google',
                    'q': 'test',
                    'api_key': serpapi_key
                }
                async with session.get('https://serpapi.com/search', params=params) as response:
                    serpapi_status = "working" if response.status == 200 else f"error: {response.status}"
        else:
            serpapi_status = "no_api_key"
    except Exception as e:
        serpapi_status = f"error: {str(e)}"
    
    # Test YouTube API
    youtube_status = "unknown"
    try:
        if youtube_key:
            async with aiohttp.ClientSession() as session:
                params = {
                    'part': 'snippet',
                    'q': 'test',
                    'type': 'video',
                    'maxResults': 1,
                    'key': youtube_key
                }
                async with session.get('https://www.googleapis.com/youtube/v3/search', params=params) as response:
                    youtube_status = "working" if response.status == 200 else f"error: {response.status}"
        else:
            youtube_status = "no_api_key"
    except Exception as e:
        youtube_status = f"error: {str(e)}"
    
    return {
        "gemini_api": gemini_status,
        "serpapi": serpapi_status,
        "youtube_api": youtube_status,
        "server_time": datetime.now().isoformat(),
        "cached_calendars": len(user_calendars),
        "endpoint_stats": endpoint_stats
    }

@app.get("/endpoint-test")
async def endpoint_test():
    """Test all endpoints to ensure they're working properly"""
    track_endpoint_usage("endpoint-test")
    logger.info("Endpoint test requested")
    
    test_results = {}
    
    # Test health endpoint
    try:
        health_response = await health_check()
        test_results["health"] = "working" if health_response["status"] == "healthy" else "error"
    except Exception as e:
        test_results["health"] = f"error: {str(e)}"
    
    # Test debug trends endpoint
    try:
        trends_response = await debug_trends("technology")
        test_results["debug_trends"] = "working" if "search_trends" in trends_response else "error"
    except Exception as e:
        test_results["debug_trends"] = f"error: {str(e)}"
    
    # Test debug model endpoint
    try:
        model_response = await debug_model()
        test_results["debug_model"] = "working" if "calendar" in model_response else "error"
    except Exception as e:
        test_results["debug_model"] = f"error: {str(e)}"
    
    # Test API status endpoint
    try:
        api_response = await api_status()
        test_results["api_status"] = "working" if "gemini_api" in api_response else "error"
    except Exception as e:
        test_results["api_status"] = f"error: {str(e)}"
    
    logger.info(f"Endpoint test completed: {test_results}")
    return {
        "endpoint_test": test_results,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/stats")
async def get_stats():
    """Get detailed statistics about API usage"""
    track_endpoint_usage("stats")
    logger.info("Statistics endpoint called")
    
    # Calculate uptime (simplified)
    uptime_seconds = (datetime.now() - app_start_time).total_seconds()
    uptime_hours = uptime_seconds / 3600
    
    return {
        "endpoint_stats": endpoint_stats,
        "uptime_seconds": uptime_seconds,
        "uptime_hours": round(uptime_hours, 2),
        "server_time": datetime.now().isoformat(),
        "cached_calendars": len(user_calendars),
        "memory_usage_mb": round(os.sys.getsizeof(user_calendars) / (1024 * 1024), 2)
    }

@app.get("/clear-cache")
async def clear_cache():
    """Clear all cached calendars"""
    track_endpoint_usage("clear-cache")
    logger.info("Clear cache endpoint called")
    
    count = len(user_calendars)
    user_calendars.clear()
    
    # Also clear dynamic content cache
    for cache_type in dynamic_content_cache:
        dynamic_content_cache[cache_type].clear()
    
    return {"status": "cache_cleared", "removed_entries": count}

# Global variable to track app start time
app_start_time = datetime.now()

@app.on_event("startup")
async def startup_event():
    logger.info("Content Platform API starting up")
    logger.info(f"Server started at: {app_start_time.isoformat()}")
    
    # Validate API keys on startup
    api_status = await validate_api_keys()
    logger.info(f"API Key Validation Results: {api_status}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Content Platform API shutting down")
    logger.info(f"Server uptime: {(datetime.now() - app_start_time).total_seconds()} seconds")

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True)
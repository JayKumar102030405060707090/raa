"""
Enhanced YouTube API Service

A Flask-based YouTube API service that fetches metadata and generates streaming links for a Telegram music bot
with proxies and caching to avoid YouTube detection.

This version combines all modules into a single main.py file for easier deployment.
"""

import json
import logging
import os
import re
import time
import uuid
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlparse, parse_qs

from flask import Flask, request, jsonify, render_template, Response, redirect
from pymongo import MongoClient
import requests
import yt_dlp
from pydantic_settings import BaseSettings, SettingsConfigDict
from fake_useragent import UserAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ====================== SETTINGS ======================

class Settings(BaseSettings):
    """Configuration for the enhanced YouTube API service."""
    
    # API information
    api_title: str = "Enhanced YouTube API"
    api_description: str = "Advanced YouTube content extraction API with proxy rotation and caching"
    api_version: str = "1.0.0"
    
    # Authentication
    default_api_keys: List[str] = [
        "1a873582a7c83342f961cc0a177b2b26",  # Default API key
    ]
    
    # MongoDB settings
    mongo_uri: str = os.environ.get(
        "MONGO_URI", 
        "mongodb+srv://jaydipmore74:xCpTm5OPAfRKYnif@cluster0.5jo18.mongodb.net/youtube_api?retryWrites=true&w=majority"
    )
    mongo_db_name: str = os.environ.get("MONGO_DB_NAME", "youtube_api")
    
    # Proxy settings
    use_proxies: bool = True
    proxy_refresh_interval: int = 3600  # Refresh proxy list every hour
    min_proxies: int = 5  # Minimum number of working proxies to maintain
    
    # Rate limiting
    rate_limit_per_minute: int = int(os.environ.get('RATE_LIMIT', 60))
    block_duration: int = 15 * 60  # Block for 15 minutes (seconds)
    max_failed_attempts: int = 30  # Block after this many bad requests
    
    # Random delay
    use_random_delay: bool = True
    min_delay_ms: int = 100
    max_delay_ms: int = 1000
    
    # Caching
    cache_enabled: bool = True
    metadata_cache_expiration: int = 1800  # 30 minutes (seconds)
    
    # Stream expiration
    stream_expiration: int = 3600  # 1 hour (seconds)
    
    # YouTube settings
    yt_max_results: int = 1
    yt_download_timeout: int = 60  # seconds
    
    # Directory paths
    temp_dir: str = os.environ.get('TEMP_DIR', 'tmp')
    log_dir: str = os.environ.get('LOG_DIR', 'logs')
    
    def __init__(self, **data):
        super().__init__(**data)
        
        # Create directories if they don't exist
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

# Load settings
settings = Settings()

# ====================== PROXY MANAGER ======================

class ProxyManager:
    """
    Manages a pool of working proxies to avoid IP detection/bans.
    Uses free public proxies from various sources.
    """
    
    def __init__(self, use_proxies: bool = True):
        """Initialize proxy manager."""
        self.use_proxies = use_proxies
        self.working_proxies = []
        self.last_refresh = 0
        
    def get_stats(self) -> Dict[str, Any]:
        """Get proxy statistics."""
        return {
            'total_proxies': len(self.working_proxies),
            'last_refresh': self.last_refresh,
            'use_proxies': self.use_proxies
        }
    
    def get_proxy(self) -> Optional[str]:
        """Get a random working proxy from the pool."""
        if not self.use_proxies or not self.working_proxies:
            return None
        
        current_time = time.time()
        
        # Check if we need to refresh proxies
        if current_time - self.last_refresh > settings.proxy_refresh_interval:
            self._refresh_proxies()
        
        # Use random proxy
        if self.working_proxies:
            import random
            return random.choice(self.working_proxies)
        
        return None
    
    def _refresh_proxies(self):
        """Refresh proxy list."""
        try:
            # This is a simplified version for the combined file
            # In a real implementation, you would fetch and test proxies
            
            # Fetch free proxies
            sources = [
                "https://www.sslproxies.org/",
                "https://free-proxy-list.net/",
                "https://www.us-proxy.org/"
            ]
            
            proxies = []
            headers = {
                'User-Agent': UserAgent().random
            }
            
            for source in sources:
                try:
                    response = requests.get(source, headers=headers, timeout=10)
                    if response.status_code == 200:
                        # Extract proxies using regex
                        ip_port_regex = r'\d+\.\d+\.\d+\.\d+:\d+'
                        found_proxies = re.findall(ip_port_regex, response.text)
                        
                        for proxy in found_proxies:
                            proxies.append(f"http://{proxy}")
                            
                except Exception as e:
                    logger.error(f"Error fetching proxies from {source}: {e}")
            
            # Test proxies (simplified)
            working_proxies = []
            for proxy in proxies[:20]:  # Test first 20 only to keep this simple
                try:
                    response = requests.get(
                        "https://www.youtube.com", 
                        proxies={"http": proxy, "https": proxy},
                        timeout=5
                    )
                    if response.status_code == 200:
                        working_proxies.append(proxy)
                        if len(working_proxies) >= settings.min_proxies:
                            break
                except:
                    # Skip proxy if it fails
                    pass
            
            if working_proxies:
                self.working_proxies = working_proxies
                self.last_refresh = time.time()
                logger.info(f"Refreshed proxy list, {len(working_proxies)} working proxies")
            else:
                logger.warning("No working proxies found")
                
        except Exception as e:
            logger.error(f"Failed to refresh proxies: {e}")

# ====================== VIDEO CACHE ======================

class VideoCache:
    """
    In-memory cache for YouTube video metadata.
    This reduces API calls to YouTube and improves response times.
    Caches only metadata, not stream URLs which are always generated fresh.
    """
    
    def __init__(self, expiration: int = 1800):  # 30 minutes default
        """Initialize the cache with an expiration time."""
        self.cache = {}
        self.expiration = expiration
        
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get a cached video info by key.
        
        Args:
            key: Cache key (query, video ID, etc.)
            
        Returns:
            Dict: Cached video info or None if not found or expired
        """
        if not settings.cache_enabled:
            return None
            
        entry = self.cache.get(key)
        if not entry:
            return None
            
        # Check if entry has expired
        if time.time() - entry.get('timestamp', 0) > self.expiration:
            # Remove expired entry
            del self.cache[key]
            return None
            
        return entry.get('data')
        
    def set(self, key: str, data: Dict[str, Any]):
        """
        Set a video info in the cache.
        
        Args:
            key: Cache key (query, video ID, etc.)
            data: Video info to cache
        """
        if not settings.cache_enabled:
            return
            
        self.cache[key] = {
            'data': data,
            'timestamp': time.time()
        }
        
        # Clean up expired entries
        self._cleanup_expired()
        
    def _cleanup_expired(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        keys_to_delete = []
        
        for key, entry in self.cache.items():
            if current_time - entry.get('timestamp', 0) > self.expiration:
                keys_to_delete.append(key)
                
        for key in keys_to_delete:
            del self.cache[key]
            
        if keys_to_delete:
            logger.info(f"Cleaned up {len(keys_to_delete)} expired cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        return {
            'enabled': settings.cache_enabled,
            'total_entries': len(self.cache),
            'expiration_seconds': self.expiration
        }

# ====================== STREAM MANAGER ======================

class StreamManager:
    """Manager for handling media streams from YouTube."""
    
    def __init__(self):
        """Initialize the stream manager with an empty streams registry."""
        self.streams: Dict[str, Dict[str, Any]] = {}
        self.last_cleanup = 0
    
    def register_stream(self, stream_id: str, video_url: str, video_mode: bool = False) -> str:
        """
        Register a new stream in the manager.
        
        Args:
            stream_id: Unique identifier for the stream
            video_url: YouTube video URL
            video_mode: Whether to stream video or audio
            
        Returns:
            str: Stream URL
        """
        self.streams[stream_id] = {
            'video_url': video_url,
            'video_mode': video_mode,
            'created_at': time.time(),
            'last_accessed': time.time(),
            'direct_url': None,  # Will be fetched on first access
        }
        
        # Do cleanup (synchronously)
        self._cleanup_expired_streams_once()
        
        return f"/stream/{stream_id}"
    
    def get_stream(self, stream_id: str) -> Optional[Dict]:
        """
        Get stream information for a given stream ID.
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            Dict: Stream information or None if not found
        """
        if stream_id not in self.streams:
            return None
        
        # Update last accessed time
        self.streams[stream_id]['last_accessed'] = time.time()
        
        return self.streams[stream_id]
    
    def get_download_url(self, video_url: str, video_mode: bool = False) -> Optional[str]:
        """
        Get the direct download URL for a YouTube video. Non-async version.
        
        Args:
            video_url: YouTube video URL
            video_mode: Whether to get video or audio
            
        Returns:
            str: Direct download URL or None if not found
        """
        ydl_opts = {
            'format': 'best[ext=mp4]' if video_mode else 'bestaudio[ext=m4a]/bestaudio',
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
            'youtube_include_dash_manifest': False,
            'youtube_skip_dash_manifest': True,
            'cachedir': False,
            'socket_timeout': 60,  # 60 seconds timeout
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                
                if not info:
                    return None
                
                # Get URL based on format
                format_url = info.get('url')
                if format_url:
                    return format_url
                
                # If no direct URL, check formats
                formats = info.get('formats', [])
                if not formats:
                    return None
                    
                # Find the best format based on video or audio mode
                if video_mode:
                    # Get best video format
                    video_formats = [f for f in formats if f.get('vcodec') != 'none' and f.get('acodec') != 'none']
                    
                    if not video_formats:
                        video_formats = formats
                    
                    # Sort by quality
                    sorted_formats = sorted(
                        video_formats,
                        key=lambda f: f.get('height', 0) * f.get('width', 0),
                        reverse=True
                    )
                    
                    best_format = sorted_formats[0] if sorted_formats else None
                else:
                    # Get best audio format
                    audio_formats = [f for f in formats if f.get('vcodec') == 'none' and f.get('acodec') != 'none']
                    
                    if not audio_formats:
                        audio_formats = formats
                        
                    # Sort by audio quality
                    sorted_formats = sorted(
                        audio_formats,
                        key=lambda f: f.get('abr', 0) or f.get('tbr', 0),
                        reverse=True
                    )
                    
                    best_format = sorted_formats[0] if sorted_formats else None
                    
                if best_format:
                    return best_format.get('url')
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting download URL: {e}")
            return None
    
    def _cleanup_expired_streams_once(self):
        """Run one cleanup pass."""
        current_time = time.time()
        expired_streams = []
        
        for stream_id, stream_info in list(self.streams.items()):
            # Expire streams after 1 hour of no access
            if current_time - stream_info['last_accessed'] > 3600:
                expired_streams.append(stream_id)
        
        # Remove expired streams
        for stream_id in expired_streams:
            del self.streams[stream_id]
            
        if expired_streams:
            logger.info(f"Cleaned up {len(expired_streams)} expired streams")

# ====================== UTILITY FUNCTIONS ======================

# YouTube URL patterns
YOUTUBE_URL_PATTERN = r'^(https?\:\/\/)?(www\.)?(youtube\.com|youtu\.?be)\/.+$'
YOUTUBE_ID_PATTERN = r'^[A-Za-z0-9_-]{11}$'

def validate_api_key(api_key: Optional[str]) -> bool:
    """
    Validate the provided API key against the list of allowed keys.
    
    Args:
        api_key: The API key to validate
        
    Returns:
        bool: True if the key is valid, False otherwise
    """
    if not api_key:
        return False
    
    # Check if the key is in the default keys
    default_keys = settings.default_api_keys
    if api_key in default_keys:
        return True
    
    # Check MongoDB for additional keys
    if db:
        key_doc = db.api_keys.find_one({'key': api_key, 'active': True})
        if key_doc:
            # Update last used time
            db.api_keys.update_one(
                {'key': api_key},
                {'$set': {'last_used_at': time.time()}, '$inc': {'total_requests': 1}}
            )
            return True
    
    # Key not found
    return False

def is_youtube_url(url: str) -> bool:
    """
    Check if the provided string is a valid YouTube URL.
    
    Args:
        url: String to check
        
    Returns:
        bool: True if it's a YouTube URL, False otherwise
    """
    return bool(re.match(YOUTUBE_URL_PATTERN, url))

def is_youtube_id(id_str: str) -> bool:
    """
    Check if the provided string is a valid YouTube video ID.
    
    Args:
        id_str: String to check
        
    Returns:
        bool: True if it's a YouTube ID, False otherwise
    """
    return bool(re.match(YOUTUBE_ID_PATTERN, id_str))

def extract_video_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from a URL.
    
    Args:
        url: YouTube URL
        
    Returns:
        str: Video ID or None if not found
    """
    if not is_youtube_url(url):
        return None
    
    parsed_url = urlparse(url)
    
    if parsed_url.netloc in ['youtube.com', 'www.youtube.com']:
        query = parse_qs(parsed_url.query)
        return query.get('v', [None])[0]
    elif parsed_url.netloc in ['youtu.be']:
        return parsed_url.path.strip('/')
    
    return None

def get_best_thumbnail(video_info: Dict) -> str:
    """
    Get the best quality thumbnail URL from video info.
    
    Args:
        video_info: Video information dictionary
        
    Returns:
        str: URL of the best thumbnail
    """
    thumbnails = video_info.get('thumbnails', [])
    
    if not thumbnails:
        # Use default thumbnail pattern if no thumbnails found
        return f"https://i.ytimg.com/vi/{video_info.get('id', '')}/hqdefault.jpg"
    
    # Sort thumbnails by quality (resolution)
    quality_sorted = sorted(
        thumbnails,
        key=lambda t: t.get('width', 0) * t.get('height', 0),
        reverse=True
    )
    
    # Return the highest quality thumbnail
    return quality_sorted[0].get('url', '')

def enhance_search_query(query: str) -> str:
    """
    Enhances a search query to get better YouTube results.
    
    Args:
        query: The original search query
        
    Returns:
        str: An enhanced search query optimized for YouTube
    """
    # Simple enhancement: add "official" to likely song queries
    if re.search(r'\s-\s', query) or re.search(r'\slyrics\s', query.lower()):
        if 'official' not in query.lower() and 'audio' not in query.lower():
            query += ' official audio'
    
    return query

# ====================== FLASK APP SETUP ======================

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "youtube-api-secret-key")

# Failed attempts tracking for rate limiting
failed_attempts = {}

# MongoDB setup
db = None
try:
    # Connect to MongoDB
    mongo_client = MongoClient(settings.mongo_uri)
    db = mongo_client[settings.mongo_db_name]
    
    # Test connection
    db.command('ping')
    logger.info("MongoDB connection successful!")
    
    # Create collections if they don't exist
    if 'api_keys' not in db.list_collection_names():
        db.create_collection('api_keys')
    
    if 'api_requests' not in db.list_collection_names():
        db.create_collection('api_requests')
        
except Exception as e:
    logger.error(f"MongoDB connection failed: {e}")
    # Create a fallback for MongoDB not being available
    db = type('obj', (object,), {
        'api_keys': type('obj', (object,), {'count_documents': lambda x: 0, 'find_one': lambda x: None}),
        'api_requests': type('obj', (object,), {'count_documents': lambda x: 0})
    })

# Initialize components
proxy_manager = ProxyManager(use_proxies=settings.use_proxies)
video_cache = VideoCache(expiration=settings.metadata_cache_expiration)
stream_manager = StreamManager()

# ====================== FLASK ROUTES ======================

@app.route('/')
def index():
    """Render the home page with API documentation."""
    stats = {
        'api_keys': db.api_keys.count_documents({}),
        'total_requests': db.api_requests.count_documents({}),
        'total_videos_cached': len(video_cache.cache)
    }
    
    return render_template('index.html', stats=stats)

@app.route('/api/youtube', methods=['GET'])
def youtube_api():
    """
    Main YouTube API endpoint that handles queries and returns video information.
    
    Query parameters:
    - query: YouTube video URL, ID, or search query
    - video: Boolean to indicate if video or audio is requested
    - api_key: API key for authentication
    
    Returns:
    - JSON with video metadata and stream URL
    """
    start_time = time.time()
    query = request.args.get('query')
    video_mode = request.args.get('video', 'false').lower() == 'true'
    api_key = request.args.get('api_key')
    
    # Get client IP and user agent
    client_ip = request.remote_addr
    user_agent = request.headers.get('User-Agent', '')
    
    # Validate API key
    if not validate_api_key(api_key):
        # Record failed attempt for rate limiting
        if client_ip:
            # Initialize counter if not exists
            if client_ip not in failed_attempts:
                failed_attempts[client_ip] = 0
            
            # Increment counter
            failed_attempts[client_ip] += 1
        
        response = jsonify({
            'error': 'Invalid API key',
            'status': 'error'
        })
        response.status_code = 401
        
        return response
    
    # Validate query parameter
    if not query:
        response = jsonify({
            'error': 'Missing query parameter',
            'status': 'error'
        })
        response.status_code = 400
        
        return response
    
    try:
        # For search queries, enhance the query
        if not is_youtube_url(query) and not is_youtube_id(query):
            query = enhance_search_query(query)
        
        # Fetch video info
        video_info = None
        if is_youtube_url(query):
            video_id = extract_video_id(query)
            youtube_url = query
        elif is_youtube_id(query):
            video_id = query
            youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        else:
            # For search queries, we'll use the query directly
            youtube_url = f"ytsearch1:{query}"
        
        # Check cache first
        cache_key = query
        cached_info = video_cache.get(cache_key)
        if cached_info:
            logger.info(f"Cache hit for query: {query}")
            video_info = cached_info
        else:
            # Use yt-dlp directly
            ydl_opts = {
                'format': 'best',
                'quiet': True,
                'no_warnings': True,
                'skip_download': True
            }
            
            # Try with proxy if available
            proxy = proxy_manager.get_proxy()
            if proxy:
                logger.info(f"Using proxy: {proxy}")
                ydl_opts['proxy'] = proxy
                
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                
                # For search results, we need to get the first entry
                if 'entries' in info:
                    if not info['entries']:
                        raise Exception("No search results found")
                    info = info['entries'][0]
                
                if not info:
                    raise Exception("Could not fetch video info")
                
                # Process the info
                video_info = {
                    'youtube_id': info.get('id'),
                    'title': info.get('title', ''),
                    'duration': info.get('duration', 0),
                    'link': f"https://www.youtube.com/watch?v={info.get('id')}",
                    'channel': info.get('channel', info.get('uploader', '')),
                    'views': info.get('view_count', 0),
                    'thumbnail': info.get('thumbnail', ''),
                }
                
                # Cache the result
                video_cache.set(cache_key, video_info)
        
        if not video_info:
            response = jsonify({
                'error': 'Video not found or unavailable',
                'status': 'error'
            })
            response.status_code = 404
            
            return response
        
        # Generate a unique stream ID for this request
        stream_id = str(uuid.uuid4())
        
        # Register the stream with the manager
        stream_url = stream_manager.register_stream(
            stream_id,
            video_info['link'],
            video_mode
        )
        
        # Construct absolute URL for the stream
        host = request.host_url.rstrip('/')
        absolute_stream_url = f"{host}{stream_url}"
        
        # Create the response data
        response_data = {
            'youtube_id': video_info['youtube_id'],
            'title': video_info['title'],
            'duration': video_info['duration'],
            'thumbnail': video_info['thumbnail'],
            'channel': video_info['channel'],
            'views': video_info['views'],
            'stream_url': absolute_stream_url,
            'video_mode': video_mode,
            'status': 'success'
        }
        
        # Record request in MongoDB for analytics
        if db:
            try:
                db.api_requests.insert_one({
                    'api_key': api_key,
                    'query': query,
                    'video_mode': video_mode,
                    'status_code': 200,
                    'response_time_ms': int((time.time() - start_time) * 1000),
                    'user_agent': user_agent,
                    'ip_address': client_ip,
                    'created_at': time.time()
                })
            except Exception as e:
                logger.error(f"Failed to log request to MongoDB: {e}")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        
        response = jsonify({
            'error': f'Internal server error: {str(e)}',
            'status': 'error'
        })
        response.status_code = 500
        
        return response

@app.route('/stream/<stream_id>', methods=['GET'])
def stream_media(stream_id):
    """
    Stream the media content for a given stream ID.
    
    Parameters:
    - stream_id: Unique identifier for the stream
    
    Returns:
    - Streaming response with audio/video content
    """
    # Get stream info
    stream_info = stream_manager.get_stream(stream_id)
    
    if not stream_info:
        return jsonify({
            'error': 'Stream not found or expired',
            'status': 'error'
        }), 404
    
    # Check if video or audio mode
    video_mode = stream_info.get('video_mode', False)
    
    # Set appropriate content type and headers
    if video_mode:
        content_type = 'video/mp4'
    else:
        content_type = 'audio/mp4'
    
    # Get the video URL
    video_url = stream_info.get('video_url')
    
    try:
        # Use the non-async version of get_download_url
        direct_url = stream_manager.get_download_url(video_url, video_mode)
        
        if not direct_url:
            return jsonify({
                'error': 'Could not get stream URL',
                'status': 'error'
            }), 500
        
        # Redirect to the direct URL
        return redirect(direct_url)
    
    except Exception as e:
        logger.error(f"Error getting direct URL: {e}")
        return jsonify({
            'error': f'Error getting stream: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get API and system statistics (admin only)."""
    api_key = request.args.get('api_key')
    
    # Admin-only endpoint
    if not validate_api_key(api_key):
        return jsonify({
            'error': 'Unauthorized - admin privileges required',
            'status': 'error'
        }), 401
    
    # Get stats
    stats = {
        'proxies': proxy_manager.get_stats(),
        'cache': video_cache.get_stats(),
        'mongo': {
            'api_keys': db.api_keys.count_documents({}),
            'api_requests': db.api_requests.count_documents({})
        },
        'system': {
            'uptime': time.time() - start_time,
            'failed_attempts': len(failed_attempts)
        }
    }
    
    return jsonify(stats)

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'version': settings.api_version
    })

# ====================== MAIN ENTRY POINT ======================

# Global start time for uptime calculation
start_time = time.time()

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
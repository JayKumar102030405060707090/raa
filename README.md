# Enhanced YouTube API

A Flask-based YouTube API service that fetches metadata and generates streaming links for Telegram music bots with proxy rotation and caching to avoid YouTube detection.

## Features

- 🎵 Fetch YouTube video metadata and generate streaming URLs
- 🔄 Proxy rotation to avoid YouTube IP bans
- 📦 Caching for improved performance and reduced API calls
- 🔑 API key authentication for security
- 🎞 Support for both audio and video streaming
- 📱 Optimized for Telegram bots
- 📊 Usage statistics and monitoring

## API Endpoints

### `/api/youtube`

Fetches metadata and generates streaming URL for a YouTube video.

**Method**: GET

**Parameters**:
- `query`: YouTube URL, video ID, or search query
- `video`: Boolean to indicate if video (true) or audio (false) is requested
- `api_key`: Your API key

**Example Request**:
```
GET /api/youtube?query=https://www.youtube.com/watch?v=dQw4w9WgXcQ&api_key=YOUR_API_KEY
```

**Example Response**:
```json
{
  "youtube_id": "dQw4w9WgXcQ",
  "title": "Rick Astley - Never Gonna Give You Up (Official Music Video)",
  "duration": 212,
  "thumbnail": "https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg",
  "channel": "Rick Astley",
  "views": 1234567890,
  "stream_url": "https://example.com/stream/uuid-here",
  "video_mode": false,
  "status": "success"
}
```

### `/stream/:stream_id`

Streams the audio/video content or redirects to the direct URL.

**Method**: GET

**Parameters**:
- `stream_id`: Stream ID returned from the YouTube API

## Deployment

### Local Development

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/youtube-api.git
   cd youtube-api
   ```

2. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements_deploy.txt
   ```

4. Set environment variables (or create a `.env` file)
   ```
   MONGO_URI=your_mongo_uri
   MONGO_DB_NAME=youtube_api
   SESSION_SECRET=your_session_secret
   ```

5. Run the application
   ```bash
   python main.py
   ```

### Heroku Deployment

1. Make sure you have the Heroku CLI installed and are logged in
   ```bash
   heroku login
   ```

2. Create a new Heroku app
   ```bash
   heroku create your-app-name
   ```

3. Set environment variables
   ```bash
   heroku config:set MONGO_URI=your_mongo_uri
   heroku config:set MONGO_DB_NAME=youtube_api
   heroku config:set SESSION_SECRET=$(openssl rand -hex 32)
   ```

4. Deploy the application
   ```bash
   git push heroku main
   ```

### Docker Deployment

1. Build the Docker image
   ```bash
   docker build -t youtube-api .
   ```

2. Run the container
   ```bash
   docker run -p 5000:5000 \
     -e MONGO_URI=your_mongo_uri \
     -e MONGO_DB_NAME=youtube_api \
     -e SESSION_SECRET=your_session_secret \
     youtube-api
   ```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGO_URI` | MongoDB connection URI | mongodb+srv://... |
| `MONGO_DB_NAME` | MongoDB database name | youtube_api |
| `SESSION_SECRET` | Secret key for Flask sessions | youtube-api-secret-key |
| `RATE_LIMIT` | Rate limit per minute | 60 |
| `TEMP_DIR` | Directory for temporary files | tmp |
| `LOG_DIR` | Directory for log files | logs |

## License

MIT License

## Acknowledgements

- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube content extraction
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Pymongo](https://pymongo.readthedocs.io/) for MongoDB integration
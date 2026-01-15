# AI Video Analysis App
Multi-Agent System for Real-Time Video & Audio Analysis

## 🎥 Overview
AI-powered video and audio analysis application using a multi-agent architecture with OpenAI GPT-4 Vision and Whisper for comprehensive real-time content analysis. 

## ✨ Features
- **Multi-Agent Pipeline**: Three specialized agents working in sequence
- **OpenAI GPT-4 Vision**: Advanced visual analysis
- **OpenAI Whisper**: High-quality audio transcription
- **Real-time Processing**: Continuous video and audio analysis
- **Priority Scoring**: Intelligent assessment and scoring (1-10)
- **Automated Reports**: Email notifications via SendGrid
- **Modern UI**: Responsive interface with real-time feedback

## 📋 Prerequisites

### System Requirements
- Python 3.8 or higher
- Modern web browser with camera and microphone access
- Stable internet connection
- At least 4GB RAM recommended

### API Keys Required
1. **OpenAI API Key** - For GPT-4 Vision and Whisper
2. **SendGrid API Key** - For email notifications

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/lizpart/Smart-Real-Time-Video-Triage-Using-AI-and-SendGrid
cd Smart-Real-Time-Video-Triage-Using-AI-and-SendGrid
```

### 2. Create Virtual Environment
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in your project root:
```env
OPENAI_API_KEY=sk-your-openai-api-key-here
SENDGRID_API_KEY=SG.your-sendgrid-api-key-here
FROM_EMAIL=noreply@yourapp.com
RECIPIENT_EMAIL=recipient@email.com
```

### 5. Verify Installation
```bash
python -c "import fastapi, openai, sendgrid; print('All dependencies installed successfully!')"
```

## 🔑 API Key Setup

### OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key and add to your `.env` file
5. **Important**: Ensure you have access to GPT-4 Vision and Whisper models

### SendGrid API Key
1. Visit [SendGrid Dashboard](https://app.sendgrid.com/)
2. Create account or sign in
3. Go to Settings > API Keys
4. Create a new API key with "Mail Send" permissions
5. Verify your sender email address in SendGrid

## 🚀 Running the Application

### Start the Server
```bash
# Make sure virtual environment is activated
python main.py

# Alternative using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Access the Application
1. Open your web browser
2. Navigate to `http://localhost:8000`
3. Allow camera and microphone access when prompted
4. Click "Start AI Analysis" to begin

## 🧪 Testing

### Health Check
Visit `http://localhost:8000/health` to verify services:
```json
{
  "status": "healthy",
  "services": {
    "openai": "connected",
    "sendgrid": "configured",
    "agents": {
      "data_capture": "active",
      "analysis": "active",
      "report": "active"
    }
  },
  "workflow": "multi-agent system operational",
  "version": "2.0.0"
}
```

### Test Analysis
1. Start analysis from the web interface
2. Speak clearly into the microphone
3. Ensure good lighting for video capture
4. Check that all three agents activate in sequence
5. Verify results appear in the analysis panel

## 🎯 Multi-Agent Workflow

### Agent 1: Data Capture Agent
- **Responsibility**: Video frame capture and audio transcription
- **Technology**: OpenAI Whisper for speech-to-text
- **Output**: Captured data with quality assessment

### Agent 2: Analysis Agent
- **Responsibility**: Visual and audio analysis with priority assessment
- **Technology**: OpenAI GPT-4 Vision (gpt-4o model)
- **Output**: Detailed analysis with scoring (1-10)

### Agent 3: Report Agent
- **Responsibility**: Report generation and email delivery
- **Technology**: HTML report generation and SendGrid
- **Output**: Formatted reports and automated email notifications

## ⚙️ Configuration Options

### Analysis Intervals
- 2 seconds (intensive)
- 5 seconds (standard) - **Default**
- 10 seconds (regular)
- 30 seconds (periodic)

### Email Alerts
Automatic emails are sent for:
- **High Priority** cases
- **Critical Priority** cases
- **Score** ≥ 7/10

### Audio Recording
- **Duration**: 3 seconds per capture
- **Format**: WebM with Opus codec

## 🐛 Troubleshooting

### Common Issues

#### "Camera Error" Status
- Ensure browser has camera permission
- Check if another application is using the camera
- Try refreshing the page

#### "Audio transcription unavailable"
- Verify OpenAI API key is correct
- Check internet connection
- Ensure microphone permissions are granted

#### "Analysis error: API request failed"
- Verify OpenAI API key and billing status
- Check if you have access to GPT-4 Vision and Whisper
- Monitor API rate limits

#### Email notifications not working
- Verify SendGrid API key and permissions
- Ensure sender email is verified in SendGrid
- Check spam folder for test emails

### Debug Mode
Enable detailed logging by setting in .env:
```env
LOG_LEVEL=DEBUG
DEBUG=True
```

## 📊 Usage Guidelines

### Best Practices
1. **Positioning**: Sit 2-3 feet from camera with good lighting
2. **Audio**: Speak clearly for accurate transcription
3. **Environment**: Use in quiet space for better audio quality
4. **Analysis**: Allow multiple analysis cycles for comprehensive assessment

### Privacy and Security
- Video and audio data is processed in real-time
- No permanent storage of media files
- API communications are encrypted
- Consider local network deployment for sensitive environments

## 🆘 Support and Maintenance

### Regular Updates
- Monitor API service status pages
- Update dependencies regularly: `pip install -r requirements.txt --upgrade`
- Check for new model versions

### Monitoring Logs
```bash
# View application logs
tail -f health_monitor.log

# Monitor in real-time
python main.py --log-level DEBUG
```

---

## � Dependencies

Core packages (see `requirements.txt`):
- `fastapi==0.116.1` - Web framework
- `openai==1.97.1` - GPT-4 Vision & Whisper
- `sendgrid==6.12.4` - Email delivery
- `python-dotenv==1.1.1` - Environment variables
- `uvicorn==0.35.0` - ASGI server
- `Jinja2==3.1.6` - Template rendering

## 🔒 Security & Privacy

- Video and audio data is processed in real-time
- No permanent storage of media files
- API communications are encrypted
- Keep API keys secure - never commit to version control
- Use environment variables for sensitive data

## 📈 Production Deployment

For production use, consider:
- Implementing rate limiting
- Adding user authentication
- Setting up database for analysis history
- Using HTTPS/SSL certificates
- Configuring proper CORS policies
- Implementing logging and monitoring

## 📄 License

See LICENSE file for details.

---

**Note**: This is an AI analysis tool. Results should be reviewed by qualified professionals for decision-making purposes.

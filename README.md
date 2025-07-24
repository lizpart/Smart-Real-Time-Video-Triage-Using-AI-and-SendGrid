# Smart Real-Time Video Triage
 - Multi-Agent System Setup Guide

## 🏥 Overview
This enhanced AI Health Monitor uses a multi-agent system powered by LangGraph, OpenAI LLM, and AssemblyAI for comprehensive real-time health assessment.

## 🚀 Features
- **Multi-Agent Architecture**: Three specialized agents working in coordination
- **OpenAI GPT-4 Vision**: Advanced visual health analysis
- **AssemblyAI Transcription**: High-quality audio-to-text conversion
- **Real-time Processing**: Continuous health monitoring
- **Priority Assessment**: Intelligent urgency scoring and risk assessment
- **Automated Alerts**: Email notifications for high-priority cases
- **Enhanced UI**: Modern, responsive interface with real-time feedback

## 📋 Prerequisites

### System Requirements
- Python 3.8 or higher
- Modern web browser with camera and microphone access
- Stable internet connection
- At least 4GB RAM recommended

### API Keys Required
1. **OpenAI API Key** - For GPT-4 Vision analysis
2. **AssemblyAI API Key** - For audio transcription
3. **SendGrid API Key** - For email notifications

## 🛠️ Installation Steps

### 1. Clone or Download the Code
```bash
# If using git
git clone <repository-url>
cd ai-health-monitor

# Or extract the provided files to a directory
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv health_monitor_env

# Activate virtual environment
# On Windows:
health_monitor_env\Scripts\activate
# On macOS/Linux:
source health_monitor_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
```bash
# Copy the template
cp .env.template .env

# Edit .env file with your API keys
nano .env  # or use your preferred editor
```

Fill in your API keys:
```env
OPENAI_API_KEY=sk-your-openai-api-key-here
ASSEMBLYAI_API_KEY=your-assemblyai-api-key-here
SENDGRID_API_KEY=SG.your-sendgrid-api-key-here
FROM_EMAIL=noreply@yourapp.com
CLINICIAN_EMAIL=doctor@hospital.com
```

### 5. Verify Installation
```bash
python -c "import fastapi, openai, assemblyai, langgraph; print('All dependencies installed successfully!')"
```

## 🔑 API Key Setup Guide

### OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key and add to your .env file
5. **Important**: Ensure you have access to GPT-4 Vision model

### AssemblyAI API Key
1. Visit [AssemblyAI Dashboard](https://www.assemblyai.com/dashboard/)
2. Sign up for an account (free tier available)
3. Navigate to "API Keys" section
4. Copy your API key and add to .env file

### SendGrid API Key
1. Visit [SendGrid Dashboard](https://app.sendgrid.com/)
2. Create account or sign in
3. Go to Settings > API Keys
4. Create a new API key with "Mail Send" permissions
5. Verify your sender email address in SendGrid

## 🚀 Running the Application

### Start the Backend Server
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
4. Click "Start AI Monitoring" to begin

## 🧪 Testing the System

### Health Check
Visit `http://localhost:8000/health` to verify all services are connected:
```json
{
  "status": "healthy",
  "services": {
    "openai": "connected",
    "assemblyai": "configured",
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
1. Start monitoring from the web interface
2. Speak clearly into the microphone (describe any symptoms)
3. Ensure good lighting for video capture
4. Check that all three agents activate in sequence
5. Verify results appear in the analysis panel

## 🎯 Multi-Agent Workflow

### Agent 1: Data Capture Agent
- **Responsibility**: Video frame capture and audio transcription
- **Technology**: AssemblyAI for speech-to-text
- **Output**: High-quality captured data with quality assessment

### Agent 2: Analysis Agent  
- **Responsibility**: Health analysis and priority assessment
- **Technology**: OpenAI GPT-4 Vision for comprehensive analysis
- **Output**: Detailed health assessment with urgency scoring

### Agent 3: Report Agent
- **Responsibility**: Report generation and communication
- **Technology**: HTML report generation and SendGrid email delivery
- **Output**: Formatted reports and automated clinician notifications

## 🔧 Configuration Options

### Monitoring Intervals
- 2 seconds (intensive monitoring)
- 5 seconds (standard monitoring) - **Default**
- 10 seconds (regular monitoring)
- 30 seconds (periodic monitoring)

### Email Alerts
Automatic emails are sent for:
- **High Priority** cases (risk assessment: high)
- **Critical Priority** cases 
- **Urgency Score** ≥ 7/10

### Audio Recording
- **Duration**: 3 seconds per capture
- **Quality**: High-fidelity with noise suppression
- **Format**: WebM with Opus codec

## 🐛 Troubleshooting

### Common Issues

#### "Camera Error" Status
- Ensure browser has camera permission
- Check if another application is using the camera
- Try refreshing the page

#### "Audio transcription unavailable"
- Verify AssemblyAI API key is correct
- Check internet connection
- Ensure microphone permissions are granted

#### "Analysis error: API request failed"
- Verify OpenAI API key and billing status
- Check if you have access to GPT-4 Vision
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

### Performance Optimization
- Close other browser tabs to free memory
- Ensure stable internet connection
- Use good lighting for better video analysis
- Speak clearly and close to microphone

## 📊 Usage Guidelines

### Best Practices
1. **Positioning**: Sit 2-3 feet from camera with good lighting
2. **Audio**: Speak clearly and describe symptoms in detail
3. **Environment**: Use in quiet space for better audio quality
4. **Monitoring**: Allow multiple analysis cycles for comprehensive assessment

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

### Contact Information
For technical support or questions about the multi-agent health monitoring system, please refer to the documentation or create an issue in the project repository.

---

## 🔒 Important Notes

⚠️ **Medical Disclaimer**: This system is for triaging and alerting purposes only. It should not replace professional medical diagnosis or treatment. Always consult healthcare professionals for medical decisions.

🔐 **Security**: Keep your API keys secure and never commit them to version control. Use environment variables and secure key management practices.

📈 **Scaling**: For production deployment, consider implementing rate limiting, user authentication, and database persistence for analysis history.

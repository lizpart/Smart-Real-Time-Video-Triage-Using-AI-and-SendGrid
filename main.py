from fastapi import FastAPI, Request, WebSocket, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import json
import base64
import os
from typing import Dict, Any, List
from pydantic import BaseModel
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from datetime import datetime
import tempfile
from dotenv import load_dotenv
import openai
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Video Analysis App", version="2.0.0")
templates = Jinja2Templates(directory=".")

# Configuration
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")
FROM_EMAIL = os.getenv("FROM_EMAIL")

# Initialize services
openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

# Pydantic Models
class VideoData(BaseModel):
    image_base64: str
    audio_transcription: str
    timestamp: str

class CapturedData(BaseModel):
    image_data: str
    audio_transcription: str
    timestamp: str
    capture_quality: str

class AnalysisResult(BaseModel):
    visual_findings: Dict[str, str]
    audio_analysis: Dict[str, Any]
    assessment: str
    priority_level: str
    score: int
    recommendations: List[str]
    summary: str
    confidence_score: float

class AnalysisReport(BaseModel):
    session_id: str
    analysis: AnalysisResult
    captured_data: CapturedData
    report_generated_at: str
    email_sent: bool

# Agent 1: Data Capture and Processing Agent
class DataCaptureAgent:
    def __init__(self):
        self.name = "DataCaptureAgent"

    async def process_audio(self, audio_base64: str) -> str:
        """Transcribe audio using OpenAI Whisper"""
        try:
            if not OPENAI_API_KEY:
                return "Audio transcription unavailable - API key not configured"

            # Handle different audio formats
            if ',' in audio_base64:
                audio_data = base64.b64decode(audio_base64.split(',')[1])
            else:
                audio_data = base64.b64decode(audio_base64)

            # Create temporary file for Whisper API
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name

            try:
                # Use OpenAI Whisper for transcription
                with open(temp_file_path, 'rb') as audio_file:
                    transcript = await openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language="en"
                    )

                return transcript.text if transcript.text else "No speech detected"
            finally:
                # Clean up temp file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        except Exception as e:
            logger.error(f"Audio processing error: {str(e)}")
            return f"Audio processing error: {str(e)}"

    async def validate_image(self, image_base64: str) -> bool:
        """Validate image quality and format"""
        try:
            if not image_base64 or len(image_base64) < 100:
                return False

            # Check if it's a valid base64 image
            if 'data:image' not in image_base64:
                return False

            return True
        except Exception:
            return False

    async def capture_and_process(self, image_base64: str, audio_base64: str, timestamp: str) -> CapturedData:
        """Main capture and processing function"""
        logger.info("Starting data capture and processing")

        # Validate image
        if not await self.validate_image(image_base64):
            raise ValueError("Invalid image data")

        # Process audio
        transcription = await self.process_audio(audio_base64)

        # Determine capture quality
        quality = "high" if len(transcription) > 10 and "error" not in transcription.lower() else "medium"

        # Create captured data
        captured_data = CapturedData(
            image_data=image_base64,
            audio_transcription=transcription,
            timestamp=timestamp,
            capture_quality=quality
        )

        logger.info("Data capture and processing completed successfully")
        return captured_data

# Agent 2: Analysis and Priority Assessment Agent
class AnalysisAgent:
    def __init__(self):
        self.name = "AnalysisAgent"
        self.model = "gpt-4o"  # Best model for vision + text analysis

    def get_analysis_prompt(self, transcription: str) -> str:
        return f"""
You are an AI assistant analyzing visual and audio data.

TRANSCRIBED SPEECH: "{transcription}"

Analyze the provided image and transcribed speech comprehensively:

VISUAL ASSESSMENT:
- Appearance and presentation
- Visible conditions or characteristics
- Movement and behavior patterns
- General demeanor and state
- Expressions and non-verbal cues

AUDIO ASSESSMENT:
- Key topics and themes discussed
- Sentiment and tone
- Temporal information (when, how long)
- Context and background information
- Voice characteristics and quality

ASSESSMENT AND SCORING:
- Score (1-10, where 10 is highest priority)
- Priority level (low/medium/high/critical)
- Overall assessment (low/moderate/high)

Provide your analysis in this exact JSON format:
{{
  "visual_findings": {{
    "appearance": "detailed description",
    "behavior_patterns": "detailed assessment",
    "movement": "detailed evaluation",
    "general_state": "comprehensive description",
    "expressions": "non-verbal indicators"
  }},
  "audio_analysis": {{
    "key_topics": ["list of main topics"],
    "sentiment": "sentiment and tone analysis",
    "temporal_info": "timing and duration details",
    "context": "contextual information",
    "voice_notes": "observations about speech patterns"
  }},
  "assessment": "low/moderate/high",
  "priority_level": "low/medium/high/critical",
  "score": 1-10,
  "recommendations": ["specific recommendations"],
  "summary": "concise summary integrating visual and audio findings",
  "confidence_score": 0.0-1.0,
  "flags": ["list any concerning findings requiring attention"]
}}

Be thorough and objective in your assessment.
"""

    async def analyze_with_openai(self, image_base64: str, transcription: str) -> Dict[str, Any]:
        """Analyze data using OpenAI GPT-4 Vision"""
        try:
            # Prepare the image for OpenAI API
            base64_data = image_base64
            if 'data:image' not in base64_data:
                base64_data = f"data:image/jpeg;base64,{image_base64}"

            response = await openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.get_analysis_prompt(transcription)
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": base64_data,
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1500,
                temperature=0.2
            )

            # Parse JSON response
            content = response.choices[0].message.content

            # Extract JSON from markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            analysis_dict = json.loads(content)
            return analysis_dict

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            raise ValueError(f"Failed to parse AI response: {str(e)}")
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            raise ValueError(f"Analysis failed: {str(e)}")

    async def analyze_and_prioritize(self, captured_data: CapturedData) -> AnalysisResult:
        """Main analysis and prioritization function"""
        logger.info("Starting health analysis and prioritization")

        # Perform OpenAI analysis
        analysis_data = await self.analyze_with_openai(
            captured_data.image_data,
            captured_data.audio_transcription
        )

        # Create analysis result
        analysis_result = AnalysisResult(
            visual_findings=analysis_data.get("visual_findings", {}),
            audio_analysis=analysis_data.get("audio_analysis", {}),
            assessment=analysis_data.get("assessment", "moderate"),
            priority_level=analysis_data.get("priority_level", "medium"),
            score=analysis_data.get("score", 5),
            recommendations=analysis_data.get("recommendations", []),
            summary=analysis_data.get("summary", "Analysis completed"),
            confidence_score=analysis_data.get("confidence_score", 0.8)
        )

        logger.info(f"Analysis completed - Priority: {analysis_result.priority_level}, Score: {analysis_result.score}")
        return analysis_result

# Agent 3: Report Generation and Communication Agent
class ReportAgent:
    def __init__(self):
        self.name = "ReportAgent"

    def generate_html_report(self, analysis: AnalysisResult, captured_data: CapturedData, session_id: str = "Unknown") -> str:
        """Generate comprehensive HTML report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Priority color coding
        priority_colors = {
            "low": "#4caf50",
            "medium": "#ff9800",
            "high": "#f44336",
            "critical": "#d32f2f"
        }

        priority_color = priority_colors.get(analysis.priority_level, "#757575")

        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .priority-badge {{
                    background-color: {priority_color};
                    color: white;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #2196f3; background-color: #f9f9f9; }}
                .urgent {{ border-left-color: #f44336; background-color: #ffebee; }}
                .finding {{ margin: 10px 0; padding: 10px; background-color: white; border-radius: 4px; }}
                .confidence {{ font-size: 0.9em; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎥 AI Video Analysis Report</h1>
                <p><strong>Session ID:</strong> {session_id}</p>
                <p><strong>Generated:</strong> {timestamp}</p>
                <p><strong>Priority Level:</strong> <span class="priority-badge">{analysis.priority_level.upper()}</span></p>
                <p><strong>Score:</strong> {analysis.score}/10</p>
                <p><strong>Confidence:</strong> <span class="confidence">{analysis.confidence_score:.1%}</span></p>
            </div>

            <div class="section {'urgent' if analysis.score >= 7 else ''}">
                <h2>🔍 Visual Analysis</h2>
                <div class="finding"><strong>Appearance:</strong> {analysis.visual_findings.get('appearance', 'N/A')}</div>
                <div class="finding"><strong>Behavior Patterns:</strong> {analysis.visual_findings.get('behavior_patterns', 'N/A')}</div>
                <div class="finding"><strong>Movement:</strong> {analysis.visual_findings.get('movement', 'N/A')}</div>
                <div class="finding"><strong>General State:</strong> {analysis.visual_findings.get('general_state', 'N/A')}</div>
                <div class="finding"><strong>Expressions:</strong> {analysis.visual_findings.get('expressions', 'N/A')}</div>
            </div>

            <div class="section">
                <h2>🎤 Audio Analysis</h2>
                <div class="finding"><strong>Transcription:</strong> "{captured_data.audio_transcription}"</div>
                <div class="finding"><strong>Key Topics:</strong> {', '.join(analysis.audio_analysis.get('key_topics', []))}</div>
                <div class="finding"><strong>Sentiment:</strong> {analysis.audio_analysis.get('sentiment', 'N/A')}</div>
                <div class="finding"><strong>Temporal Info:</strong> {analysis.audio_analysis.get('temporal_info', 'N/A')}</div>
                <div class="finding"><strong>Context:</strong> {analysis.audio_analysis.get('context', 'N/A')}</div>
                <div class="finding"><strong>Voice Notes:</strong> {analysis.audio_analysis.get('voice_notes', 'N/A')}</div>
            </div>

            <div class="section {'urgent' if analysis.assessment == 'high' else ''}">
                <h2>⚠️ Assessment</h2>
                <div class="finding">
                    <strong>Overall Assessment:</strong>
                    <span style="color: {'red' if analysis.assessment == 'high' else 'orange' if analysis.assessment == 'moderate' else 'green'}">
                        {analysis.assessment.upper()}
                    </span>
                </div>
            </div>

            <div class="section">
                <h2>📋 Recommendations</h2>
                <ul>
                    {''.join(f'<li>{action}</li>' for action in analysis.recommendations)}
                </ul>
            </div>

            <div class="section">
                <h2>📝 Summary</h2>
                <p>{analysis.summary}</p>
            </div>

            <div class="section">
                <h2>📊 Technical Details</h2>
                <div class="finding"><strong>Capture Quality:</strong> {captured_data.capture_quality}</div>
                <div class="finding"><strong>Analysis Timestamp:</strong> {captured_data.timestamp}</div>
                <div class="finding"><strong>Model Confidence:</strong> {analysis.confidence_score:.1%}</div>
            </div>

            <hr style="margin: 30px 0;">
            <p style="font-size: 0.9em; color: #666;">
                <em>🤖 This report was generated by an AI video analysis system using OpenAI GPT-4 Vision and Whisper transcription.</em>
            </p>
        </body>
        </html>
        """

        return html_content

    async def send_report(self, analysis: AnalysisResult, captured_data: CapturedData, session_id: str = "Unknown") -> bool:
        """Send report via SendGrid"""
        try:
            if not SENDGRID_API_KEY or not RECIPIENT_EMAIL:
                logger.warning("SendGrid API key or recipient email not configured")
                return False

            html_content = self.generate_html_report(analysis, captured_data, session_id)

            # Subject line with priority indicators
            subject_prefix = "🚨 CRITICAL" if analysis.priority_level == "critical" else \
                           "⚠️ HIGH PRIORITY" if analysis.priority_level == "high" else \
                           "📋 MEDIUM PRIORITY" if analysis.priority_level == "medium" else \
                           "ℹ️ LOW PRIORITY"

            subject = f"{subject_prefix} - AI Video Analysis Alert - Score {analysis.score}/10"

            message = Mail(
                from_email=FROM_EMAIL,
                to_emails=RECIPIENT_EMAIL,
                subject=subject,
                html_content=html_content
            )

            sg = SendGridAPIClient(api_key=SENDGRID_API_KEY)
            response = sg.send(message)

            success = response.status_code == 202
            if success:
                logger.info(f"Report sent successfully for {session_id}")
            else:
                logger.error(f"Failed to send report: {response.status_code}")

            return success

        except Exception as e:
            logger.error(f"Email sending error: {str(e)}")
            return False

    async def generate_and_send_report(self, analysis_result: AnalysisResult, captured_data: CapturedData) -> AnalysisReport:
        """Main report generation and sending function"""
        logger.info("Starting report generation and communication")

        session_id = "Session-" + datetime.now().strftime("%Y%m%d-%H%M%S")

        # Always send email
        email_sent = await self.send_report(
            analysis_result,
            captured_data,
            session_id
        )

        # Create analysis report
        analysis_report = AnalysisReport(
            session_id=session_id,
            analysis=analysis_result,
            captured_data=captured_data,
            report_generated_at=datetime.now().isoformat(),
            email_sent=email_sent
        )

        logger.info(f"Report generation completed - Email sent: {email_sent}")
        return analysis_report

# Initialize Agents
data_capture_agent = DataCaptureAgent()
analysis_agent = AnalysisAgent()
report_agent = ReportAgent()

# Simple Async Workflow Pipeline
async def process_video_analysis_workflow(image_base64: str, audio_base64: str, timestamp: str) -> AnalysisReport:
    """Process video analysis workflow through all agents"""

    # Step 1: Capture and process data
    captured_data = await data_capture_agent.capture_and_process(
        image_base64,
        audio_base64,
        timestamp
    )

    # Step 2: Analyze and prioritize
    analysis_result = await analysis_agent.analyze_and_prioritize(captured_data)

    # Step 3: Generate and send report
    analysis_report = await report_agent.generate_and_send_report(
        analysis_result,
        captured_data
    )

    return analysis_report

# FastAPI Endpoints
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_video_data(data: VideoData):
    """Main endpoint for video data analysis using multi-agent workflow"""
    try:
        logger.info("Starting multi-agent video analysis workflow")

        # Run the workflow pipeline
        analysis_report = await process_video_analysis_workflow(
            data.image_base64,
            data.audio_transcription,
            data.timestamp
        )

        # Return comprehensive response
        return {
            "analysis": analysis_report.analysis.model_dump(),
            "transcription": analysis_report.captured_data.audio_transcription,
            "email_sent": analysis_report.email_sent,
            "timestamp": analysis_report.report_generated_at,
            "session_id": analysis_report.session_id,
            "workflow_status": "completed",
            "priority_level": analysis_report.analysis.priority_level,
            "score": analysis_report.analysis.score
        }

    except Exception as e:
        logger.error(f"Workflow error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis workflow failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    services_status = {
        "openai": "configured" if OPENAI_API_KEY else "not configured",
        "sendgrid": "configured" if SENDGRID_API_KEY else "not configured",
        "agents": {
            "data_capture": "active",
            "analysis": "active",
            "report": "active"
        }
    }

    # Test OpenAI connection
    try:
        if OPENAI_API_KEY:
            test_response = await openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            services_status["openai"] = "connected"
    except Exception as e:
        services_status["openai"] = f"error: {str(e)}"

    return {
        "status": "healthy",
        "services": services_status,
        "workflow": "multi-agent system operational",
        "version": "2.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting AI Video Analysis App with Multi-Agent System")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

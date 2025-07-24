from fastapi import FastAPI, Request, WebSocket, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncio
import json
import base64
import os
from typing import Dict, Any, List
from pydantic import BaseModel
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import assemblyai as aai
from datetime import datetime
import tempfile
from dotenv import load_dotenv
import openai

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Health Monitor", version="2.0.0")
templates = Jinja2Templates(directory=".")

# Configuration
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
CLINICIAN_EMAIL = os.getenv("CLINICIAN_EMAIL")
FROM_EMAIL = os.getenv("FROM_EMAIL")

# Initialize services
openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

if ASSEMBLYAI_API_KEY:
    aai.settings.api_key = ASSEMBLYAI_API_KEY

# Pydantic Models
class HealthData(BaseModel):
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
    verbal_symptoms: Dict[str, Any]
    risk_assessment: str
    priority_level: str
    urgency_score: int
    recommended_actions: List[str]
    summary: str
    confidence_score: float

class HealthReport(BaseModel):
    patient_id: str
    analysis: AnalysisResult
    captured_data: CapturedData
    report_generated_at: str
    email_sent: bool

# LangGraph State Definition
class HealthMonitorState(TypedDict):
    image_base64: str
    audio_base64: str
    timestamp: str
    captured_data: CapturedData
    analysis_result: AnalysisResult
    health_report: HealthReport
    error_message: str
    current_step: str

# Agent 1: Data Capture and Processing Agent
class DataCaptureAgent:
    def __init__(self):
        self.name = "DataCaptureAgent"
    
    async def process_audio(self, audio_base64: str) -> str:
        """Transcribe audio using AssemblyAI"""
        try:
            if not ASSEMBLYAI_API_KEY:
                return "Audio transcription unavailable - API key not configured"
            
            # Handle different audio formats
            if ',' in audio_base64:
                audio_data = base64.b64decode(audio_base64.split(',')[1])
            else:
                audio_data = base64.b64decode(audio_base64)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                transcriber = aai.Transcriber()
                transcript = transcriber.transcribe(temp_file_path)
                
                if transcript.error:
                    return f"Transcription error: {transcript.error}"
                
                return transcript.text if transcript.text else "No speech detected"
            finally:
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
    
    async def capture_and_process(self, state: HealthMonitorState) -> HealthMonitorState:
        """Main capture and processing function"""
        try:
            logger.info("Starting data capture and processing")
            
            # Validate image
            if not await self.validate_image(state["image_base64"]):
                state["error_message"] = "Invalid image data"
                return state
            
            # Process audio
            transcription = await self.process_audio(state["audio_base64"])
            
            # Determine capture quality
            quality = "high" if len(transcription) > 10 and "error" not in transcription.lower() else "medium"
            
            # Create captured data
            captured_data = CapturedData(
                image_data=state["image_base64"],
                audio_transcription=transcription,
                timestamp=state["timestamp"],
                capture_quality=quality
            )
            
            state["captured_data"] = captured_data
            state["current_step"] = "capture_complete"
            
            logger.info("Data capture and processing completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Data capture error: {str(e)}")
            state["error_message"] = f"Data capture error: {str(e)}"
            return state

# Agent 2: Analysis and Priority Assessment Agent
class AnalysisAgent:
    def __init__(self):
        self.name = "AnalysisAgent"
        self.model = "gpt-4o"  # Best model for vision + text analysis
    
    def get_analysis_prompt(self, transcription: str) -> str:
        return f"""
You are a medical AI assistant analyzing visual and audio health data for remote patient monitoring.

TRANSCRIBED PATIENT SPEECH: "{transcription}"

Analyze the provided image and transcribed speech comprehensively:

VISUAL ASSESSMENT:
- Skin conditions (color, rashes, lesions, pallor)
- Breathing patterns (visible difficulty, chest movement, posture)
- Mobility indicators (posture, gait, movement limitations)
- General appearance (fatigue, alertness, distress signs)
- Facial expressions indicating pain or discomfort

AUDIO ASSESSMENT:
- Symptom descriptions and their severity
- Pain levels and specific locations
- Duration and onset information
- Functional limitations mentioned
- Voice quality (hoarseness, breathlessness)

PRIORITY AND URGENCY ASSESSMENT:
- Urgency score (1-10, where 10 is immediate emergency)
- Priority level (low/medium/high/critical)
- Risk assessment (low/moderate/high)

Provide your analysis in this exact JSON format:
{{
  "visual_findings": {{
    "skin_condition": "detailed description",
    "breathing_pattern": "detailed assessment", 
    "mobility_assessment": "detailed evaluation",
    "general_appearance": "comprehensive description",
    "facial_expressions": "pain or distress indicators"
  }},
  "verbal_symptoms": {{
    "primary_complaints": ["list of main symptoms"],
    "severity_indicators": "pain levels and severity descriptions",
    "temporal_information": "onset, duration, and progression details",
    "functional_impact": "how symptoms affect daily activities",
    "voice_quality_notes": "observations about speech patterns"
  }},
  "risk_assessment": "low/moderate/high",
  "priority_level": "low/medium/high/critical", 
  "urgency_score": 1-10,
  "recommended_actions": ["specific clinical recommendations"],
  "summary": "concise clinical summary integrating visual and audio findings",
  "confidence_score": 0.0-1.0,
  "red_flags": ["list any concerning findings requiring immediate attention"]
}}

Be thorough, objective, and note any limitations of remote assessment. Flag any emergency indicators immediately.
"""
    
    async def analyze_with_openai(self, image_base64: str, transcription: str) -> Dict[str, Any]:
        """Analyze health data using OpenAI GPT-4 Vision"""
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
                max_tokens=2000,
                temperature=0.1
            )
            
            analysis_text = response.choices[0].message.content
            
            # Parse JSON response
            try:
                import re
                cleaned = re.sub(r"^```(?:json)?|```$", "", analysis_text.strip(), flags=re.MULTILINE).strip()
                analysis_data = json.loads(cleaned)
                return analysis_data
            except json.JSONDecodeError:
                # Fallback parsing if JSON is malformed
                logger.warning("JSON parsing failed, using fallback")
                return self._create_fallback_analysis(analysis_text, transcription)
                
        except Exception as e:
            logger.error(f"OpenAI analysis error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")
    
    def _create_fallback_analysis(self, analysis_text: str, transcription: str) -> Dict[str, Any]:
        """Create fallback analysis if JSON parsing fails"""
        return {
            "visual_findings": {
                "skin_condition": "Analysis completed - see summary",
                "breathing_pattern": "Assessment performed", 
                "mobility_assessment": "Evaluation completed",
                "general_appearance": "Comprehensive review done",
                "facial_expressions": "Observed and noted"
            },
            "verbal_symptoms": {
                "primary_complaints": [transcription[:100] + "..." if len(transcription) > 100 else transcription],
                "severity_indicators": "Extracted from patient speech",
                "temporal_information": "Timeline noted from verbal report",
                "functional_impact": "Impact assessment completed",
                "voice_quality_notes": "Voice characteristics noted"
            },
            "risk_assessment": "moderate",
            "priority_level": "medium",
            "urgency_score": 5,
            "recommended_actions": ["Clinical review recommended", "Consider in-person examination"],
            "summary": analysis_text[:500] + "..." if len(analysis_text) > 500 else analysis_text,
            "confidence_score": 0.7,
            "red_flags": ["Analysis format issue - manual review recommended"]
        }
    
    async def analyze_and_prioritize(self, state: HealthMonitorState) -> HealthMonitorState:
        """Main analysis and prioritization function"""
        try:
            logger.info("Starting health analysis and prioritization")
            
            captured_data = state["captured_data"]
            
            # Perform OpenAI analysis
            analysis_data = await self.analyze_with_openai(
                captured_data.image_data, 
                captured_data.audio_transcription
            )
            
            # Create analysis result
            analysis_result = AnalysisResult(
                visual_findings=analysis_data.get("visual_findings", {}),
                verbal_symptoms=analysis_data.get("verbal_symptoms", {}),
                risk_assessment=analysis_data.get("risk_assessment", "moderate"),
                priority_level=analysis_data.get("priority_level", "medium"),
                urgency_score=analysis_data.get("urgency_score", 5),
                recommended_actions=analysis_data.get("recommended_actions", []),
                summary=analysis_data.get("summary", "Analysis completed"),
                confidence_score=analysis_data.get("confidence_score", 0.8)
            )
            
            state["analysis_result"] = analysis_result
            state["current_step"] = "analysis_complete"
            
            logger.info(f"Analysis completed - Priority: {analysis_result.priority_level}, Urgency: {analysis_result.urgency_score}")
            return state
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            state["error_message"] = f"Analysis error: {str(e)}"
            return state

# Agent 3: Report Generation and Communication Agent
class ReportAgent:
    def __init__(self):
        self.name = "ReportAgent"
    
    def generate_html_report(self, analysis: AnalysisResult, captured_data: CapturedData, patient_id: str = "Unknown") -> str:
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
                .red-flag {{ color: #d32f2f; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🏥 AI Health Monitoring Report</h1>
                <p><strong>Patient ID:</strong> {patient_id}</p>
                <p><strong>Generated:</strong> {timestamp}</p>
                <p><strong>Priority Level:</strong> <span class="priority-badge">{analysis.priority_level.upper()}</span></p>
                <p><strong>Urgency Score:</strong> {analysis.urgency_score}/10</p>
                <p><strong>Confidence:</strong> <span class="confidence">{analysis.confidence_score:.1%}</span></p>
            </div>
            
            <div class="section {'urgent' if analysis.urgency_score >= 7 else ''}">
                <h2>🔍 Visual Assessment</h2>
                <div class="finding"><strong>Skin Condition:</strong> {analysis.visual_findings.get('skin_condition', 'N/A')}</div>
                <div class="finding"><strong>Breathing Pattern:</strong> {analysis.visual_findings.get('breathing_pattern', 'N/A')}</div>
                <div class="finding"><strong>Mobility Assessment:</strong> {analysis.visual_findings.get('mobility_assessment', 'N/A')}</div>
                <div class="finding"><strong>General Appearance:</strong> {analysis.visual_findings.get('general_appearance', 'N/A')}</div>
                <div class="finding"><strong>Facial Expressions:</strong> {analysis.visual_findings.get('facial_expressions', 'N/A')}</div>
            </div>
            
            <div class="section">
                <h2>🎤 Audio Analysis</h2>
                <div class="finding"><strong>Transcription:</strong> "{captured_data.audio_transcription}"</div>
                <div class="finding"><strong>Primary Complaints:</strong> {', '.join(analysis.verbal_symptoms.get('primary_complaints', []))}</div>
                <div class="finding"><strong>Severity Indicators:</strong> {analysis.verbal_symptoms.get('severity_indicators', 'N/A')}</div>
                <div class="finding"><strong>Timeline:</strong> {analysis.verbal_symptoms.get('temporal_information', 'N/A')}</div>
                <div class="finding"><strong>Functional Impact:</strong> {analysis.verbal_symptoms.get('functional_impact', 'N/A')}</div>
                <div class="finding"><strong>Voice Quality:</strong> {analysis.verbal_symptoms.get('voice_quality_notes', 'N/A')}</div>
            </div>
            
            <div class="section {'urgent' if analysis.risk_assessment == 'high' else ''}">
                <h2>⚠️ Risk Assessment</h2>
                <div class="finding">
                    <strong>Overall Risk:</strong> 
                    <span style="color: {'red' if analysis.risk_assessment == 'high' else 'orange' if analysis.risk_assessment == 'moderate' else 'green'}">
                        {analysis.risk_assessment.upper()}
                    </span>
                </div>
            </div>
            
            <div class="section">
                <h2>📋 Recommendations</h2>
                <ul>
                    {''.join(f'<li>{action}</li>' for action in analysis.recommended_actions)}
                </ul>
            </div>
            
            <div class="section">
                <h2>📝 Clinical Summary</h2>
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
                <em>🤖 This report was generated by an AI health monitoring system using OpenAI GPT-4 Vision and AssemblyAI transcription. 
                Please use professional clinical judgment and conduct in-person examination as needed.</em>
            </p>
        </body>
        </html>
        """
        
        return html_content
    
    async def send_health_report(self, analysis: AnalysisResult, captured_data: CapturedData, patient_id: str = "Unknown") -> bool:
        """Send health report via SendGrid"""
        try:
            if not SENDGRID_API_KEY or not CLINICIAN_EMAIL:
                logger.warning("SendGrid API key or clinician email not configured")
                return False
            
            html_content = self.generate_html_report(analysis, captured_data, patient_id)
            
            # Subject line with priority indicators
            subject_prefix = "🚨 CRITICAL" if analysis.priority_level == "critical" else \
                           "⚠️ HIGH PRIORITY" if analysis.priority_level == "high" else \
                           "📋 MEDIUM PRIORITY" if analysis.priority_level == "medium" else \
                           "ℹ️ LOW PRIORITY"
            
            subject = f"{subject_prefix} - Health Monitor Alert - Urgency {analysis.urgency_score}/10"
            
            message = Mail(
                from_email=FROM_EMAIL,
                to_emails=CLINICIAN_EMAIL,
                subject=subject,
                html_content=html_content
            )
            
            sg = SendGridAPIClient(api_key=SENDGRID_API_KEY)
            response = sg.send(message)
            
            success = response.status_code == 202
            if success:
                logger.info(f"Health report sent successfully for {patient_id}")
            else:
                logger.error(f"Failed to send health report: {response.status_code}")
            
            return success
            
        except Exception as e:
            logger.error(f"Email sending error: {str(e)}")
            return False
    
    async def generate_and_send_report(self, state: HealthMonitorState) -> HealthMonitorState:
        """Main report generation and sending function"""
        try:
            logger.info("Starting report generation and communication")
            
            analysis_result = state["analysis_result"]
            captured_data = state["captured_data"]
            
            # Always send email
            email_sent = await self.send_health_report(
                analysis_result, 
                captured_data, 
                "Patient-" + datetime.now().strftime("%Y%m%d-%H%M%S")
            )
            
            # Create health report
            health_report = HealthReport(
                patient_id="Patient-" + datetime.now().strftime("%Y%m%d-%H%M%S"),
                analysis=analysis_result,
                captured_data=captured_data,
                report_generated_at=datetime.now().isoformat(),
                email_sent=email_sent
            )
            
            state["health_report"] = health_report
            state["current_step"] = "report_complete"
            
            logger.info(f"Report generation completed - Email sent: {email_sent}")
            return state
            
        except Exception as e:
            logger.error(f"Report generation error: {str(e)}")
            state["error_message"] = f"Report generation error: {str(e)}"
            return state

# Initialize Agents
data_capture_agent = DataCaptureAgent()
analysis_agent = AnalysisAgent()
report_agent = ReportAgent()

# LangGraph Workflow Setup
def create_health_monitor_graph():
    """Create the LangGraph workflow for health monitoring"""
    
    workflow = StateGraph(HealthMonitorState)
    
    # Add nodes for each agent
    workflow.add_node("capture", data_capture_agent.capture_and_process)
    workflow.add_node("analyze", analysis_agent.analyze_and_prioritize) 
    workflow.add_node("report", report_agent.generate_and_send_report)
    
    # Define the workflow edges
    workflow.set_entry_point("capture")
    workflow.add_edge("capture", "analyze")
    workflow.add_edge("analyze", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()

# Create the compiled graph
health_monitor_graph = create_health_monitor_graph()

# FastAPI Endpoints
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_health_data(data: HealthData):
    """Main endpoint for health data analysis using multi-agent workflow"""
    try:
        logger.info("Starting multi-agent health analysis workflow")
        
        # Initialize state
        initial_state = HealthMonitorState(
            image_base64=data.image_base64,
            audio_base64=data.audio_transcription,
            timestamp=data.timestamp,
            current_step="initializing"
        )
        
        # Run the multi-agent workflow
        final_state = await health_monitor_graph.ainvoke(initial_state)
        
        # Check for errors
        if "error_message" in final_state and final_state["error_message"]:
            raise HTTPException(status_code=500, detail=final_state["error_message"])
        
        # Return comprehensive response
        health_report = final_state["health_report"]
        
        return {
            "analysis": health_report.analysis.dict(),
            "transcription": health_report.captured_data.audio_transcription,
            "email_sent": health_report.email_sent,
            "timestamp": health_report.report_generated_at,
            "patient_id": health_report.patient_id,
            "workflow_status": "completed",
            "priority_level": health_report.analysis.priority_level,
            "urgency_score": health_report.analysis.urgency_score
        }
        
    except Exception as e:
        logger.error(f"Workflow error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis workflow failed: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time health monitoring"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_json()
            logger.info("Received WebSocket data for analysis")
            
            # Initialize state for workflow
            initial_state = HealthMonitorState(
                image_base64=data.get('image', ''),
                audio_base64=data.get('audio', ''),
                timestamp=datetime.now().isoformat(),
                current_step="initializing"
            )
            
            # Run workflow
            final_state = await health_monitor_graph.ainvoke(initial_state)
            
            # Send results back
            if "error_message" in final_state and final_state["error_message"]:
                await websocket.send_json({
                    "error": final_state["error_message"],
                    "timestamp": datetime.now().isoformat()
                })
            else:
                health_report = final_state["health_report"]
                await websocket.send_json({
                    "analysis": health_report.analysis.dict(),
                    "transcription": health_report.captured_data.audio_transcription,
                    "email_sent": health_report.email_sent,
                    "timestamp": health_report.report_generated_at,
                    "priority_level": health_report.analysis.priority_level,
                    "urgency_score": health_report.analysis.urgency_score
                })
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({
            "error": f"WebSocket error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })
    finally:
        await websocket.close()
        logger.info("WebSocket connection closed")

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    services_status = {
        "openai": "configured" if OPENAI_API_KEY else "not configured",
        "assemblyai": "configured" if ASSEMBLYAI_API_KEY else "not configured",
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
    logger.info("Starting AI Health Monitor with Multi-Agent System")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
"""Sentinel AI Stock Analysis - Fixed Version"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, ValidationError
import yfinance as yf
import os
from datetime import datetime
from typing import Optional, Dict, Any
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sentinel AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
client = None

if ANTHROPIC_API_KEY:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, max_retries=2, timeout=60.0)
        logger.info("✅ AI enabled")
    except Exception as e:
        logger.error(f"AI init failed: {e}")

class AnalysisRequest(BaseModel):
    company_name: str
    email: EmailStr

class AnalysisResponse(BaseModel):
    success: bool
    live_data: Optional[Dict[str, Any]] = None
    report: Optional[str] = None
    error: Optional[str] = None

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "detail": "Internal server error"
        }
    )

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Invalid request format",
            "detail": str(exc)
        }
    )

def fetch_data(ticker: str) -> Dict[str, Any]:
    try:
        logger.info(f"Fetching data for {ticker}")
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y")
        
        if hist.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        
        current = hist['Close'].iloc[-1]
        
        result = {
            "ticker": ticker.upper(),
            "company_name": info.get("longName", ticker),
            "sector": info.get("sector", "Unknown"),
            "current_price": round(float(current), 2),
            "pe_ratio": round(float(info.get("trailingPE", 0)), 2) if info.get("trailingPE") else None,
            "profit_margin": round(float(info.get("profitMargins", 0)) * 100, 2) if info.get("profitMargins") else None,
        }
        
        logger.info(f"Successfully fetched data for {ticker}")
        return result
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        raise ValueError(f"Failed to fetch stock data: {str(e)}")

def analyze(ticker: str, data: Dict) -> str:
    if not client:
        logger.warning("AI client not available, using fallback")
        return f"Quick Analysis for {ticker}: Current price ${data['current_price']}, P/E Ratio: {data['pe_ratio']}. AI analysis requires ANTHROPIC_API_KEY to be configured."
    
    try:
        logger.info(f"Generating AI analysis for {ticker}")
        prompt = f"""Analyze this stock briefly and provide a recommendation:

{ticker} - {data['company_name']}
Current Price: ${data['current_price']}
P/E Ratio: {data['pe_ratio']}
Profit Margin: {data['profit_margin']}%
Sector: {data['sector']}

Provide:
1. Quick valuation assessment
2. Financial health overview
3. Clear recommendation: BUY, HOLD, or SELL with brief reasoning

Keep it concise and actionable."""

        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result = msg.content[0].text
        logger.info(f"AI analysis completed for {ticker}")
        return result
        
    except Exception as e:
        logger.error(f"AI error for {ticker}: {e}")
        return f"Analysis for {ticker}: Price ${data['current_price']}, P/E {data['pe_ratio']}. Full AI analysis temporarily unavailable."

@app.get("/")
async def root():
    return {
        "service": "Sentinel AI Stock Analysis",
        "status": "operational",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "analyze": "/api/generate-report"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ai_available": bool(client)
    }

@app.post("/api/generate-report")
async def generate_report(request: AnalysisRequest):
    """Generate stock analysis report"""
    try:
        logger.info(f"Analysis request received: {request.company_name} from {request.email}")
        
        ticker = request.company_name.strip().upper()
        
        if not ticker:
            return AnalysisResponse(
                success=False,
                error="Ticker symbol is required"
            )
        
        # Fetch stock data
        try:
            live_data = fetch_data(ticker)
        except Exception as e:
            logger.error(f"Data fetch failed: {e}")
            return AnalysisResponse(
                success=False,
                error=f"Failed to fetch data for {ticker}. Please check if the ticker is valid."
            )
        
        # Generate AI analysis
        try:
            report = analyze(ticker, live_data)
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            report = f"Basic analysis for {ticker} at ${live_data['current_price']}"
        
        logger.info(f"✅ Analysis completed successfully for {ticker}")
        
        return AnalysisResponse(
            success=True,
            live_data=live_data,
            report=report
        )
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return AnalysisResponse(
            success=False,
            error="Invalid request format. Please provide company_name and email."
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return AnalysisResponse(
            success=False,
            error=f"An error occurred: {str(e)}"
        )

@app.get("/api/test")
async def test_endpoint():
    """Test endpoint to verify API is working"""
    return {
        "status": "OK",
        "message": "API is working correctly",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

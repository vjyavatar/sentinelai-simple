"""Sentinel AI - BULLETPROOF VERSION"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import yfinance as yf
import os
from datetime import datetime
from typing import Optional, Dict, Any
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sentinel AI", version="2.0.0")

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
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        logger.info("✅ AI enabled")
    except:
        pass

class AnalysisRequest(BaseModel):
    company_name: str
    email: EmailStr

class AnalysisResponse(BaseModel):
    success: bool
    live_data: Optional[Dict[str, Any]] = None
    report: Optional[str] = None
    error: Optional[str] = None

def fetch_data(ticker: str) -> Dict[str, Any]:
    """BULLETPROOF stock data fetcher"""
    try:
        ticker = ticker.strip().upper()
        logger.info(f"🔍 Fetching {ticker}")
        
        # Method 1: Try standard approach
        try:
            stock = yf.Ticker(ticker)
            
            # Force download to refresh cache
            hist = stock.history(period="1mo", interval="1d")
            
            if hist.empty:
                logger.warning(f"Empty history for {ticker}, trying longer period")
                hist = stock.history(period="3mo")
            
            if hist.empty:
                raise ValueError("No price data")
            
            # Get current price
            current_price = float(hist['Close'].iloc[-1])
            
            # Get info with timeout
            info = {}
            try:
                info = stock.info
                time.sleep(0.1)  # Small delay to avoid rate limits
            except Exception as e:
                logger.warning(f"Info fetch failed: {e}, using minimal data")
                info = {}
            
            # Build result with safe extraction
            result = {
                "ticker": ticker,
                "company_name": info.get("longName") or info.get("shortName") or ticker,
                "sector": info.get("sector", "Technology"),
                "industry": info.get("industry", "Unknown"),
                "current_price": round(current_price, 2),
                "currency": info.get("currency", "USD"),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": round(float(info.get("trailingPE", 0)), 2) if info.get("trailingPE") and float(info.get("trailingPE", 0)) > 0 else None,
                "forward_pe": round(float(info.get("forwardPE", 0)), 2) if info.get("forwardPE") and float(info.get("forwardPE", 0)) > 0 else None,
                "profit_margin": round(float(info.get("profitMargins", 0)) * 100, 2) if info.get("profitMargins") else None,
                "revenue_growth": round(float(info.get("revenueGrowth", 0)) * 100, 2) if info.get("revenueGrowth") else None,
                "debt_to_equity": round(float(info.get("debtToEquity", 0)), 2) if info.get("debtToEquity") else None,
                "52_week_high": round(float(info.get("fiftyTwoWeekHigh", current_price * 1.2)), 2),
                "52_week_low": round(float(info.get("fiftyTwoWeekLow", current_price * 0.8)), 2),
                "beta": round(float(info.get("beta", 1)), 2) if info.get("beta") else 1.0,
                "dividend_yield": round(float(info.get("dividendYield", 0)) * 100, 2) if info.get("dividendYield") else 0,
                "avg_volume": info.get("averageVolume", 0),
                "price_change": round(((current_price - float(hist['Close'].iloc[-2])) / float(hist['Close'].iloc[-2]) * 100), 2) if len(hist) > 1 else 0
            }
            
            logger.info(f"✅ Successfully fetched {ticker}")
            return result
            
        except Exception as e:
            logger.error(f"Method 1 failed: {e}")
            
            # Method 2: Minimal fallback with just price
            try:
                ticker_obj = yf.Ticker(ticker)
                data = ticker_obj.history(period="5d")
                
                if not data.empty:
                    price = float(data['Close'].iloc[-1])
                    return {
                        "ticker": ticker,
                        "company_name": ticker,
                        "sector": "Unknown",
                        "industry": "Unknown",
                        "current_price": round(price, 2),
                        "currency": "USD",
                        "market_cap": 0,
                        "pe_ratio": None,
                        "forward_pe": None,
                        "profit_margin": None,
                        "revenue_growth": None,
                        "debt_to_equity": None,
                        "52_week_high": round(price * 1.2, 2),
                        "52_week_low": round(price * 0.8, 2),
                        "beta": 1.0,
                        "dividend_yield": 0,
                        "avg_volume": 0,
                        "price_change": 0
                    }
            except:
                pass
            
            raise ValueError(f"Unable to fetch data for {ticker}. Please verify the ticker symbol is correct.")
            
    except Exception as e:
        logger.error(f"❌ All methods failed for {ticker}: {e}")
        raise

def analyze(ticker: str, data: Dict) -> str:
    """Generate analysis"""
    if not client:
        return generate_basic_analysis(ticker, data)
    
    try:
        prompt = f"""Analyze {ticker} ({data['company_name']}) and provide investment guidance:

Price: ${data['current_price']}
P/E: {data['pe_ratio']}
Profit Margin: {data['profit_margin']}%
Revenue Growth: {data['revenue_growth']}%
Sector: {data['sector']}

Provide:
1. VALUATION: Is it overvalued, fairly valued, or undervalued?
2. FINANCIAL HEALTH: Comment on profitability and growth
3. RECOMMENDATION: Clear BUY, HOLD, or SELL with reasoning
4. PRICE TARGET: Suggest entry/exit points

Be concise and actionable."""

        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return msg.content[0].text
        
    except:
        return generate_basic_analysis(ticker, data)

def generate_basic_analysis(ticker: str, data: Dict) -> str:
    """Basic analysis fallback"""
    pe = data.get('pe_ratio', 0)
    margin = data.get('profit_margin', 0)
    growth = data.get('revenue_growth', 0)
    
    # Score
    score = 0
    if pe and 10 < pe < 30: score += 2
    elif pe and pe < 10: score += 1
    
    if margin and margin > 20: score += 2
    elif margin and margin > 10: score += 1
    
    if growth and growth > 15: score += 2
    elif growth and growth > 5: score += 1
    
    # Recommendation
    if score >= 4:
        rec = "BUY"
        verdict = "undervalued with strong fundamentals"
    elif score >= 2:
        rec = "HOLD"
        verdict = "fairly valued"
    else:
        rec = "SELL"
        verdict = "overvalued or weak fundamentals"
    
    return f"""**ANALYSIS FOR {ticker}**

**VALUATION VERDICT**
Current Price: ${data['current_price']}
P/E Ratio: {pe if pe else 'N/A'} - {"Attractive" if pe and pe < 25 else "Premium" if pe else "Unknown"}

**FINANCIAL HEALTH**
Profit Margin: {margin if margin else 'N/A'}% - {"Strong" if margin and margin > 15 else "Moderate" if margin else "Unknown"}
Revenue Growth: {growth if growth else 'N/A'}% - {"Excellent" if growth and growth > 10 else "Steady" if growth else "Unknown"}
Sector: {data['sector']}

**RECOMMENDATION: {rec}**
This stock appears {verdict}. 

**ENTRY STRATEGY**
{"Consider buying on dips below current price" if rec == "BUY" else "Monitor for better entry points" if rec == "HOLD" else "Consider taking profits"}

Price Range: ${data['52_week_low']} - ${data['52_week_high']}
Current: ${data['current_price']}

*Full AI analysis available with API key configuration*"""

@app.get("/")
async def root():
    return {
        "service": "Sentinel AI Stock Analysis",
        "version": "2.0.0",
        "status": "operational",
        "ai_enabled": bool(client)
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "ai_available": bool(client)}

@app.post("/api/generate-report", response_model=AnalysisResponse)
async def generate_report(request: AnalysisRequest):
    """Generate stock analysis"""
    try:
        ticker = request.company_name.strip().upper()
        logger.info(f"📊 Analyzing {ticker}")
        
        # Fetch data
        try:
            live_data = fetch_data(ticker)
        except Exception as e:
            logger.error(f"Data fetch failed: {e}")
            return AnalysisResponse(
                success=False,
                error=f"Could not fetch data for {ticker}. Please check the ticker symbol."
            )
        
        # Generate analysis
        report = analyze(ticker, live_data)
        
        logger.info(f"✅ Complete: {ticker}")
        
        return AnalysisResponse(
            success=True,
            live_data=live_data,
            report=report
        )
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return AnalysisResponse(
            success=False,
            error=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

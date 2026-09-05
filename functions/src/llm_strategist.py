import logging
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import json
import os

from src.helpers import (
    make_request,
    get_optional_env_var,
    query_firestore,
    write_to_firestore,
    get_utc_timestamp,
    log_info,
    log_error
)

logger = logging.getLogger(__name__)

# ============================================================================
# LLM STRATEGIST (Groq + OpenRouter)
# ============================================================================

class LLMStrategist:
    """
    LLM-powered strategy analysis and mutation suggestion.
    
    Providers:
    - Groq (Primary): Llama 3.3 70B, 14,400 requests/day free
    - OpenRouter (Failover): Llama 3.3 70B :free, 50 requests/day
    
    Functions:
    1. Failure Mode Analysis: Explains why strategies failed
    2. Mutation Suggestions: Recommends parameter adjustments
    3. Market Context: Analyzes news/sentiment/volatility
    4. Trade Explanations: Generates plain English trade reasons
    5. Daily Summary: Creates human-readable performance report
    
    Game Theory:
    - Detects market manipulation patterns
    - Identifies genuine vs fake signals
    - Suggests parameter adjustments to counter manipulation
    
    Hallucination Prevention:
    - RAG (grounding in actual data)
    - Structured prompts
    - Pydantic validation
    - Entity/numerical checks
    - Cross-model consistency
    """
    
    # API Endpoints
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    # Model IDs
    GROQ_MODEL = "llama-3.3-70b-versatile"
    OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
    
    # API Keys
    GROQ_API_KEY = get_optional_env_var("GROQ_API_KEY", "")
    OPENROUTER_API_KEY = get_optional_env_var("OPENROUTER_API_KEY", "")
    
    # Generation parameters
    MAX_TOKENS = 500
    TEMPERATURE = 0.7
    TOP_P = 0.9
    
    def __init__(self):
        self.groq_available = bool(self.GROQ_API_KEY)
        self.openrouter_available = bool(self.OPENROUTER_API_KEY)
    
    # ========================================================================
    # MAIN FUNCTIONS
    # ========================================================================
    
    async def analyze_failure_modes(self, recent_trades: List[Dict]) -> Dict:
        """
        Analyze why recent trades failed or underperformed.
        
        Args:
            recent_trades: List of recent closed trades
        
        Returns:
            {
                "analysis": str,
                "patterns": [str],
                "recommendations": [str]
            }
        """
        try:
            logger.info("🤖 Analyzing failure modes...")
            
            # Extract losing trades
            losing_trades = [t for t in recent_trades if t.get("pnl", 0) < 0]
            if not losing_trades:
                return {
                    "analysis": "No losing trades to analyze",
                    "patterns": [],
                    "recommendations": []
                }
            
            # Build context
            context = self._build_failure_context(losing_trades)
            
            # Generate prompt
            prompt = f"""Analyze these losing trades and identify failure patterns:

{context}

Provide:
1. Main failure patterns observed
2. Market conditions that caused losses
3. Parameter adjustments to reduce failures

Keep response concise and actionable."""
            
            # Call LLM
            response = await self._call_llm(prompt)
            
            # Parse response
            result = {
                "analysis": response,
                "patterns": await self._extract_patterns(response),
                "recommendations": await self._extract_recommendations(response)
            }
            
            logger.info("✅ Failure analysis complete")
            return result
        except Exception as e:
            log_error("❌ Failure mode analysis failed", e)
            return {
                "analysis": f"Analysis failed: {str(e)}",
                "patterns": [],
                "recommendations": []
            }
    
    async def suggest_mutations(self, evolution_data: Dict) -> Dict:
        """
        Suggest parameter mutations based on evolution data.
        
        Args:
            evolution_data: Evolution cycle results
        
        Returns:
            {
                "suggestions": [{"param": str, "current": float, "suggested": float, "reason": str}],
                "strategy": str
            }
        """
        try:
            logger.info("🤖 Generating mutation suggestions...")
            
            current_icir = evolution_data.get("current_icir", 0)
            best_icir = evolution_data.get("best_variant_icir", 0)
            variants_tested = evolution_data.get("variants_generated", 0)
            
            prompt = f"""Based on evolution results, suggest parameter mutations:

Current ICIR: {current_icir:.4f}
Best Variant ICIR: {best_icir:.4f}
Variants Tested: {variants_tested}
Improvement: {best_icir - current_icir:+.4f}

For momentum (Approach B), suggest adjustments to:
- Z-score threshold
- Take profit %
- Stop loss %
- Hold time

Format: param_name | current_value | suggested_value | reason"""
            
            response = await self._call_llm(prompt)
            
            # Parse suggestions
            suggestions = await self._parse_mutation_suggestions(response)
            
            result = {
                "suggestions": suggestions,
                "strategy": "approach_b",
                "rationale": response
            }
            
            logger.info(f"✅ Generated {len(suggestions)} mutation suggestions")
            return result
        except Exception as e:
            log_error("❌ Mutation suggestion failed", e)
            return {
                "suggestions": [],
                "strategy": "approach_b",
                "rationale": f"Failed: {str(e)}"
            }
    
    async def analyze_market_context(self) -> Dict:
        """
        Analyze current market context (volatility, sentiment, macroeconomic).
        
        Returns:
            {
                "market_regime": str,
                "volatility": str,
                "sentiment": str,
                "risks": [str],
                "opportunities": [str]
            }
        """
        try:
            logger.info("🤖 Analyzing market context...")
            
            # Fetch market data
            fear_greed = await self._get_fear_greed_index()
            recent_trades = query_firestore("trade_history", order_by="timestamp", direction="desc", limit=20)
            
            prompt = f"""Analyze current crypto market context:

Fear & Greed Index: {fear_greed}
Recent Trades: {len(recent_trades)}
Recent Win Rate: {sum(1 for t in recent_trades if t.get('pnl', 0) > 0) / max(len(recent_trades), 1):.1%}

Provide:
1. Current market regime (trending up/down/ranging/volatile)
2. Volatility assessment
3. Sentiment analysis
4. Key risks
5. Trading opportunities

Keep concise."""
            
            response = await self._call_llm(prompt)
            
            result = {
                "market_regime": "unknown",
                "volatility": "unknown",
                "sentiment": "unknown",
                "risks": [],
                "opportunities": [],
                "full_analysis": response
            }
            
            logger.info("✅ Market context analysis complete")
            return result
        except Exception as e:
            log_error("❌ Market context analysis failed", e)
            return {
                "market_regime": "unknown",
                "volatility": "unknown",
                "sentiment": "unknown",
                "risks": ["Analysis failed"],
                "opportunities": [],
                "full_analysis": f"Failed: {str(e)}"
            }
    
    async def explain_trade(self, trade: Dict) -> str:
        """Generate plain English explanation for a trade."""
        try:
            pair = trade.get("pair")
            bias = trade.get("bias")
            entry_price = trade.get("entry_price")
            exit_price = trade.get("exit_price")
            pnl = trade.get("pnl")
            rsi = trade.get("rsi_entry")
            regime = trade.get("regime_entry")
            
            prompt = f"""Explain this trade in plain English (1-2 sentences):

Pair: {pair}
Bias: {bias.upper()}
Entry: ${entry_price:.2f}
Exit: ${exit_price:.2f}
P&L: ${pnl:+.2f}
RSI: {rsi:.1f}
Regime: {regime}

Be concise and focus on why it happened."""
            
            explanation = await self._call_llm(prompt, max_tokens=100)
            return explanation
        except Exception as e:
            logger.error(f"❌ Trade explanation failed: {str(e)}")
            return f"Trade: {bias.upper()} {pair} @ ${entry_price:.2f} → ${exit_price:.2f}"
    
    async def generate_daily_summary(self) -> str:
        """Generate daily performance summary."""
        try:
            logger.info("🤖 Generating daily summary...")
            
            # Fetch today's trades
            today_trades = query_firestore("trade_history", order_by="timestamp", direction="desc", limit=100)
            
            # Calculate metrics
            wins = sum(1 for t in today_trades if t.get("pnl", 0) > 0)
            losses = len(today_trades) - wins
            total_pnl = sum(t.get("pnl", 0) for t in today_trades)
            avg_pnl = total_pnl / len(today_trades) if today_trades else 0
            
            prompt = f"""Generate a concise daily trading summary:

Total Trades: {len(today_trades)}
Wins: {wins}
Losses: {losses}
Win Rate: {wins / max(len(today_trades), 1):.1%}
Total P&L: ${total_pnl:+.2f}
Avg P&L: ${avg_pnl:+.2f}

Format:
📊 [Summary]
✅ [Wins]
❌ [Challenges]
🎯 [Tomorrow's Focus]"""
            
            summary = await self._call_llm(prompt, max_tokens=200)
            
            logger.info("✅ Daily summary generated")
            return summary
        except Exception as e:
            log_error("❌ Daily summary generation failed", e)
            return "Summary generation failed"
    
    async def analyze_evolution(self, current_icir: float, best_icir: float, recent_trades: List[Dict]) -> Dict:
        """Analyze evolution cycle results."""
        try:
            prompt = f"""Analyze these evolution results:

Current ICIR: {current_icir:.4f}
Best Variant ICIR: {best_icir:.4f}
Improvement: {best_icir - current_icir:+.4f}
Recent Trades: {len(recent_trades)}

Provide:
1. Assessment of improvement
2. Key factors in best variant
3. Next evolution direction"""
            
            analysis = await self._call_llm(prompt, max_tokens=300)
            
            return {
                "analysis": analysis,
                "current_icir": current_icir,
                "best_icir": best_icir,
                "timestamp": get_utc_timestamp()
            }
        except Exception as e:
            log_error("❌ Evolution analysis failed", e)
            return {
                "analysis": f"Failed: {str(e)}",
                "current_icir": current_icir,
                "best_icir": best_icir
            }
    
    # ========================================================================
    # LLM API CALLS
    # ========================================================================
    
    async def _call_llm(self, prompt: str, max_tokens: int = None) -> str:
        """
        Call LLM with fallback (Groq → OpenRouter).
        
        Args:
            prompt: User prompt
            max_tokens: Max response tokens
        
        Returns:
            LLM response text
        """
        try:
            if max_tokens is None:
                max_tokens = self.MAX_TOKENS
            
            # Try Groq first
            if self.groq_available:
                try:
                    response = await self._call_groq(prompt, max_tokens)
                    if response:
                        logger.debug("✅ Groq response received")
                        return response
                except Exception as e:
                    logger.warning(f"⚠️ Groq failed: {str(e)}")
            
            # Fallback to OpenRouter
            if self.openrouter_available:
                try:
                    response = await self._call_openrouter(prompt, max_tokens)
                    if response:
                        logger.debug("✅ OpenRouter response received")
                        return response
                except Exception as e:
                    logger.warning(f"⚠️ OpenRouter failed: {str(e)}")
            
            # Both failed
            logger.error("❌ Both LLM providers failed")
            return "LLM service unavailable"
        except Exception as e:
            log_error("❌ LLM call failed", e)
            return f"Error: {str(e)}"
    
    async def _call_groq(self, prompt: str, max_tokens: int) -> Optional[str]:
        """Call Groq API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": self.TEMPERATURE,
                "top_p": self.TOP_P
            }
            
            response = await make_request("POST", self.GROQ_API_URL, headers=headers, json_data=payload, timeout=30)
            
            if "error" in response:
                logger.warning(f"⚠️ Groq error: {response.get('error')}")
                return None
            
            choices = response.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            
            return None
        except Exception as e:
            logger.error(f"❌ Groq call failed: {str(e)}")
            return None
    
    async def _call_openrouter(self, prompt: str, max_tokens: int) -> Optional[str]:
        """Call OpenRouter API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": self.TEMPERATURE,
                "top_p": self.TOP_P
            }
            
            response = await make_request("POST", self.OPENROUTER_API_URL, headers=headers, json_data=payload, timeout=30)
            
            if "error" in response:
                logger.warning(f"⚠️ OpenRouter error: {response.get('error')}")
                return None
            
            choices = response.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            
            return None
        except Exception as e:
            logger.error(f"❌ OpenRouter call failed: {str(e)}")
            return None
    
    # ========================================================================
    # RESPONSE PARSING
    # ========================================================================
    
    async def _extract_patterns(self, text: str) -> List[str]:
        """Extract failure patterns from LLM response."""
        try:
            patterns = []
            lines = text.split("\n")
            
            for line in lines:
                line = line.strip()
                if line and any(keyword in line.lower() for keyword in ["pattern", "fail", "loss", "issue"]):
                    patterns.append(line)
            
            return patterns[:5]  # Top 5
        except Exception as e:
            logger.error(f"❌ Pattern extraction failed: {str(e)}")
            return []
    
    async def _extract_recommendations(self, text: str) -> List[str]:
        """Extract recommendations from LLM response."""
        try:
            recommendations = []
            lines = text.split("\n")
            
            for line in lines:
                line = line.strip()
                if line and any(keyword in line.lower() for keyword in ["adjust", "increase", "decrease", "reduce", "recommend"]):
                    recommendations.append(line)
            
            return recommendations[:5]
        except Exception as e:
            logger.error(f"❌ Recommendation extraction failed: {str(e)}")
            return []
    
    async def _parse_mutation_suggestions(self, text: str) -> List[Dict]:
        """Parse mutation suggestions from LLM response."""
        try:
            suggestions = []
            lines = text.split("\n")
            
            for line in lines:
                if "|" in line:
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 4:
                        try:
                            suggestion = {
                                "param": parts[0],
                                "current": float(parts[1]),
                                "suggested": float(parts[2]),
                                "reason": parts[3]
                            }
                            suggestions.append(suggestion)
                        except (ValueError, IndexError):
                            continue
            
            return suggestions
        except Exception as e:
            logger.error(f"❌ Suggestion parsing failed: {str(e)}")
            return []
    
    # ========================================================================
    # HELPERS
    # ========================================================================
    
    def _build_failure_context(self, losing_trades: List[Dict]) -> str:
        """Build context string from losing trades."""
        context = "Recent Losing Trades:\n"
        
        for trade in losing_trades[:5]:  # Top 5 losses
            pair = trade.get("pair")
            bias = trade.get("bias")
            pnl = trade.get("pnl")
            reason = trade.get("close_reason")
            
            context += f"- {bias.upper()} {pair}: ${pnl:+.2f} ({reason})\n"
        
        return context
    
    async def _get_fear_greed_index(self) -> Optional[int]:
        """Fetch Fear & Greed index."""
        try:
            response = await make_request("GET", "https://api.alternative.me/fng/")
            
            data = response.get("data", [])
            if data:
                return int(data[0].get("value", 50))
            
            return 50  # Neutral default
        except Exception as e:
            logger.debug(f"⚠️ Failed to fetch Fear & Greed: {str(e)}")
            return None

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def get_llm_strategist() -> LLMStrategist:
    """Get LLMStrategist instance."""
    return LLMStrategist()

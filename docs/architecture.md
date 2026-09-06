# Kraken Sentinel V15.0 - System Architecture

**Version:** 15.0  
**Last Updated:** September 2026  
**Author:** Belinda  
**License:** MIT  

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Layers](#architecture-layers)
3. [Component Breakdown](#component-breakdown)
4. [Data Flow](#data-flow)
5. [Firestore Schema](#firestore-schema)
6. [Strategy Mechanics](#strategy-mechanics)
7. [Risk Management](#risk-management)
8. [Evolution Engine](#evolution-engine)
9. [LLM Integration](#llm-integration)
10. [Dashboard Architecture](#dashboard-architecture)
11. [Deployment Model](#deployment-model)

---

## System Overview

Kraken Sentinel V15.0 is a **fully autonomous cryptocurrency trading bot** running on Firebase Cloud Functions with Firestore for real-time state management. It executes two parallel strategies (Momentum & Swing) on Kraken spot markets with zero-cost LLM decision support.

### Key Attributes

| Aspect | Detail |
|--------|--------|
| **Platform** | Firebase Cloud Functions (Python 3.11) |
| **Database** | Firestore (real-time, event-driven) |
| **Exchange** | Kraken REST API (spot trading, Phase 1) |
| **Strategies** | Momentum (2-6h holds) + Swing (4-24h holds) |
| **Capital** | $300 Phase 1 ($150/strategy), scales to $1,390 over 7 phases |
| **LLM** | Groq (Llama 3.3 70B) + OpenRouter failover |
| **Risk Model** | ATR stops (2x), Kelly Criterion (25% conservative), daily loss -5%, max drawdown -10% |
| **Validation** | ICIR (>0.2) + Walk-Forward (Spearman correlation) |
| **Dashboard** | Real-time Firestore updates, Chart.js, cyberpunk UI |
| **Cost** | $0/month (free tiers: Kraken, CoinPaprika, alternative.me, Groq) |

---

## Architecture Layers

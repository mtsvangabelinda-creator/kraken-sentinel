import logging
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time

from src.helpers import (
    make_request,
    write_to_firestore,
    query_firestore,
    get_utc_timestamp,
    log_info,
    log_error,
    get_firestore_client,
    retry_with_backoff
)

logger = logging.getLogger(__name__)

# ============================================================================
# OHLCV FETCHER
# ============================================================================

class OHLCVFetcher:
    """
    Fetch OHLCV data from Kraken API and store in Firestore.
    
    Data sources:
    - Kraken Public API (free, no key required)
    
    Intervals supported:
    - 1m: 1-minute candles
    - 5m: 5-minute candles
    - 1h: 1-hour candles
    - 1d: 1-day candles
    
    Retention:
    - Store last 100 candles per interval per pair
    - Daily archival of historical data
    """
    
    # Kraken API
    KRAKEN_API_URL = "https://api.kraken.com/0/public/OHLC"
    KRAKEN_ASSETS_URL = "https://api.kraken.com/0/public/AssetPairs"
    
    # Intervals (in minutes)
    INTERVALS = {
        "1": "1m",
        "5": "5m",
        "60": "1h",
        "1440": "1d"
    }
    
    # Limits per API call
    MAX_CANDLES_PER_REQUEST = 720  # Kraken limit
    
    # Firestore
    OHLCV_COLLECTION = "ohlcv_data"
    HISTORICAL_COLLECTION = "ohlcv_historical"
    
    def __init__(self):
        self.db = get_firestore_client()
    
    async def fetch_all_ohlcv(self) -> Dict[str, Dict]:
        """
        Fetch OHLCV data for all Kraken pairs.
        
        Returns:
            Dict of {pair: ohlcv_data}
        """
        try:
            logger.info("📊 Starting OHLCV fetch for all pairs...")
            
            # Fetch all pairs
            pairs = await self._fetch_all_pairs()
            logger.info(f"📈 Fetching OHLCV for {len(pairs)} pairs")
            
            all_ohlcv = {}
            
            # Fetch in parallel (batch of 20)
            for i in range(0, len(pairs), 20):
                batch = pairs[i:i+20]
                batch_results = await asyncio.gather(*[
                    self._fetch_pair_ohlcv(pair) for pair in batch
                ], return_exceptions=True)
                
                for pair, ohlcv_data in zip(batch, batch_results):
                    if isinstance(ohlcv_data, dict) and ohlcv_data:
                        all_ohlcv[pair] = ohlcv_data
                
                # Rate limit: Kraken has strict API limits
                await asyncio.sleep(1)
            
            logger.info(f"✅ Fetched OHLCV for {len(all_ohlcv)} pairs")
            
            # Save to Firestore
            await self._save_all_ohlcv(all_ohlcv)
            
            return all_ohlcv
        except Exception as e:
            log_error("❌ OHLCV fetch failed", e)
            raise
    
    async def _fetch_pair_ohlcv(self, pair: str) -> Optional[Dict]:
        """
        Fetch OHLCV data for a single pair across all intervals.
        
        Returns:
            Dict of {interval: ohlcv_list}
        """
        try:
            ohlcv_data = {}
            
            # Fetch each interval
            for interval_key, interval_name in self.INTERVALS.items():
                try:
                    candles = await self._fetch_interval_ohlcv(pair, interval_key, limit=100)
                    if candles:
                        ohlcv_data[interval_name] = candles
                except Exception as e:
                    logger.warning(f"⚠️ Failed to fetch {pair} {interval_name}: {str(e)}")
                    continue
                
                # Rate limiting between intervals
                await asyncio.sleep(0.1)
            
            if ohlcv_data:
                logger.info(f"✅ Fetched {pair}: {len(ohlcv_data)} intervals")
                return ohlcv_data
            else:
                return None
        except Exception as e:
            logger.error(f"❌ Failed to fetch {pair} OHLCV: {str(e)}")
            return None
    
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def _fetch_interval_ohlcv(self, pair: str, interval: str, limit: int = 100) -> List[Dict]:
        """
        Fetch OHLCV data for specific interval from Kraken.
        
        Args:
            pair: Trading pair (e.g., "BTCUSD")
            interval: Time interval in minutes (e.g., "60" for 1 hour)
            limit: Number of candles to fetch
        
        Returns:
            List of OHLCV dicts
        """
        try:
            params = {
                "pair": pair,
                "interval": interval,
                "limit": min(limit, self.MAX_CANDLES_PER_REQUEST)
            }
            
            response = await make_request("GET", self.KRAKEN_API_URL, params=params, timeout=15)
            
            # Check for Kraken API errors
            if "error" in response and response["error"]:
                error_msg = response["error"][0] if response["error"] else "Unknown error"
                logger.warning(f"⚠️ Kraken API error for {pair}: {error_msg}")
                return []
            
            # Parse response
            result = response.get("result", {})
            if pair not in result:
                logger.warning(f"⚠️ No data for {pair}")
                return []
            
            ohlcv_raw = result[pair]
            
            # Convert to structured format
            ohlcv_list = [
                {
                    "time": int(candle[0]),
                    "timestamp": datetime.utcfromtimestamp(candle[0]).isoformat(),
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "vwap": float(candle[5]),
                    "volume": float(candle[6]),
                    "count": int(candle[7])
                }
                for candle in ohlcv_raw
            ]
            
            logger.debug(f"✅ Fetched {len(ohlcv_list)} candles for {pair} interval {interval}")
            return ohlcv_list
        except Exception as e:
            logger.error(f"❌ Failed to fetch {pair} interval {interval}: {str(e)}")
            raise
    
    # ========================================================================
    # SAVE TO FIRESTORE
    # ========================================================================
    
    async def _save_all_ohlcv(self, all_ohlcv: Dict[str, Dict]) -> bool:
        """Save all OHLCV data to Firestore."""
        try:
            timestamp = get_utc_timestamp()
            
            for pair, ohlcv_data in all_ohlcv.items():
                # Save current data
                doc_id = f"{pair}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                data = {
                    "pair": pair,
                    "timestamp": timestamp,
                    "data": ohlcv_data
                }
                
                write_to_firestore(self.OHLCV_COLLECTION, doc_id, data)
            
            logger.info(f"✅ Saved OHLCV for {len(all_ohlcv)} pairs to Firestore")
            return True
        except Exception as e:
            log_error("❌ Failed to save OHLCV to Firestore", e)
            return False
    
    # ========================================================================
    # HISTORICAL DATA ARCHIVE
    # ========================================================================
    
    async def archive_historical_data(self, pair: str, days: int = 365) -> bool:
        """
        Archive historical OHLCV data for backtesting.
        
        Fetches N days of 1-day candles for a pair.
        
        Args:
            pair: Trading pair
            days: Number of days to fetch
        
        Returns:
            True if successful
        """
        try:
            logger.info(f"📚 Archiving {days} days of historical data for {pair}...")
            
            # Calculate number of requests needed (720 candles per request max)
            requests_needed = (days // 720) + 1
            all_candles = []
            
            for request_num in range(requests_needed):
                offset = request_num * 720
                limit = min(720, days - offset)
                
                if limit <= 0:
                    break
                
                candles = await self._fetch_interval_ohlcv(pair, "1440", limit=limit)
                all_candles.extend(candles)
                
                # Rate limiting
                await asyncio.sleep(0.5)
            
            if not all_candles:
                logger.warning(f"⚠️ No historical data for {pair}")
                return False
            
            # Save to Firestore
            doc_id = f"{pair}_{datetime.utcnow().strftime('%Y%m%d')}"
            data = {
                "pair": pair,
                "timestamp": get_utc_timestamp(),
                "days_archived": days,
                "candle_count": len(all_candles),
                "candles": all_candles
            }
            
            write_to_firestore(self.HISTORICAL_COLLECTION, doc_id, data)
            
            logger.info(f"✅ Archived {len(all_candles)} candles for {pair}")
            return True
        except Exception as e:
            log_error(f"❌ Failed to archive historical data for {pair}", e)
            return False
    
    # ========================================================================
    # RETRIEVE FROM FIRESTORE
    # ========================================================================
    
    def get_latest_ohlcv(self, pair: str, interval: str = "1h") -> Optional[List[Dict]]:
        """
        Retrieve latest OHLCV data from Firestore.
        
        Args:
            pair: Trading pair
            interval: Time interval ("1m", "5m", "1h", "1d")
        
        Returns:
            List of OHLCV candles or None
        """
        try:
            docs = query_firestore(
                self.OHLCV_COLLECTION,
                field="pair",
                operator="==",
                value=pair,
                order_by="timestamp",
                direction="desc",
                limit=1
            )
            
            if not docs:
                logger.warning(f"⚠️ No OHLCV found for {pair}")
                return None
            
            latest_doc = docs[0]
            ohlcv_data = latest_doc.get("data", {})
            
            return ohlcv_data.get(interval, [])
        except Exception as e:
            log_error(f"❌ Failed to retrieve OHLCV for {pair}", e)
            return None
    
    def get_historical_ohlcv(self, pair: str, days: int = 365) -> Optional[List[Dict]]:
        """
        Retrieve historical OHLCV data from Firestore.
        
        Args:
            pair: Trading pair
            days: Days to retrieve
        
        Returns:
            List of historical candles or None
        """
        try:
            docs = query_firestore(
                self.HISTORICAL_COLLECTION,
                field="pair",
                operator="==",
                value=pair,
                order_by="timestamp",
                direction="desc",
                limit=1
            )
            
            if not docs:
                logger.warning(f"⚠️ No historical data for {pair}")
                return None
            
            return docs[0].get("candles", [])
        except Exception as e:
            log_error(f"❌ Failed to retrieve historical OHLCV for {pair}", e)
            return None
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    async def _fetch_all_pairs(self) -> List[str]:
        """Fetch all available trading pairs from Kraken."""
        try:
            response = await make_request("GET", self.KRAKEN_ASSETS_URL)
            
            if "error" in response and response["error"]:
                logger.error(f"❌ Failed to fetch pairs: {response['error']}")
                return []
            
            pairs = [key for key in response.get("result", {}).keys() if not key.startswith("X")]
            logger.info(f"✅ Fetched {len(pairs)} trading pairs")
            
            return pairs
        except Exception as e:
            log_error("❌ Failed to fetch all pairs", e)
            return []
    
    async def validate_pair(self, pair: str) -> bool:
        """Validate that a pair exists on Kraken."""
        try:
            params = {"pair": pair, "interval": "60", "limit": 1}
            response = await make_request("GET", self.KRAKEN_API_URL, params=params)
            
            if "error" in response and response["error"]:
                return False
            
            return pair in response.get("result", {})
        except Exception as e:
            logger.error(f"❌ Failed to validate pair {pair}: {str(e)}")
            return False
    
    def calculate_candle_time_remaining(self, interval: str) -> int:
        """
        Calculate seconds until next candle closes.
        
        Args:
            interval: Time interval ("1m", "5m", "1h", "1d")
        
        Returns:
            Seconds until next candle
        """
        try:
            interval_map = {
                "1m": 60,
                "5m": 300,
                "1h": 3600,
                "1d": 86400
            }
            
            interval_seconds = interval_map.get(interval, 3600)
            
            now = datetime.utcnow()
            epoch = now.timestamp()
            seconds_into_candle = epoch % interval_seconds
            
            return int(interval_seconds - seconds_into_candle)
        except Exception as e:
            logger.error(f"❌ Failed to calculate candle time: {str(e)}")
            return 60
    
    def get_time_until_next_candle(self, interval: str) -> timedelta:
        """Get timedelta until next candle closes."""
        seconds = self.calculate_candle_time_remaining(interval)
        return timedelta(seconds=seconds)

# ============================================================================
# MAIN FUNCTION
# ============================================================================

async def fetch_ohlcv():
    """Main entry point for OHLCV fetcher."""
    try:
        fetcher = OHLCVFetcher()
        all_ohlcv = await fetcher.fetch_all_ohlcv()
        return all_ohlcv
    except Exception as e:
        log_error("❌ OHLCV fetch failed", e)
        raise

async def archive_historical_ohlcv(pair: str, days: int = 365):
    """Main entry point for historical archival."""
    try:
        fetcher = OHLCVFetcher()
        success = await fetcher.archive_historical_data(pair, days)
        return success
    except Exception as e:
        log_error(f"❌ Historical archive failed for {pair}", e)
        raise

import os
import logging
import asyncio
import aiohttp
import firebase_admin
from firebase_admin import firestore, credentials
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from functools import wraps
import json
import pytz

logger = logging.getLogger(__name__)

# ============================================================================
# FIREBASE INITIALIZATION
# ============================================================================

def get_firestore_client():
    """Get Firestore client instance."""
    try:
        db = firestore.client()
        return db
    except Exception as e:
        logger.error(f"❌ Failed to get Firestore client: {str(e)}")
        raise

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================

def get_env_var(var_name: str, default: Optional[str] = None) -> str:
    """
    Get environment variable safely.
    Raises error if required var is missing.
    """
    value = os.getenv(var_name, default)
    if value is None:
        raise ValueError(f"❌ Required environment variable '{var_name}' not set")
    return value

def get_optional_env_var(var_name: str, default: str = None) -> Optional[str]:
    """Get optional environment variable."""
    return os.getenv(var_name, default)

# ============================================================================
# RETRY LOGIC (Exponential Backoff)
# ============================================================================

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """
    Decorator for exponential backoff retry logic.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (doubles each retry)
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"❌ Max retries reached for {func.__name__}: {str(e)}")
                        raise
                    
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"⚠️ Retry {attempt + 1}/{max_retries} for {func.__name__} after {delay}s delay. Error: {str(e)}")
                    await asyncio.sleep(delay)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"❌ Max retries reached for {func.__name__}: {str(e)}")
                        raise
                    
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"⚠️ Retry {attempt + 1}/{max_retries} for {func.__name__} after {delay}s delay. Error: {str(e)}")
                    asyncio.sleep(delay)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# ============================================================================
# HTTP REQUESTS WITH RETRY
# ============================================================================

async def make_request(
    method: str,
    url: str,
    headers: Optional[Dict] = None,
    json_data: Optional[Dict] = None,
    params: Optional[Dict] = None,
    timeout: int = 10,
    max_retries: int = 3
) -> Dict:
    """
    Make HTTP request with exponential backoff retry.
    
    Args:
        method: HTTP method (GET, POST, etc.)
        url: Request URL
        headers: Request headers
        json_data: JSON body
        params: Query parameters
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
    
    Returns:
        Response data as dict
    """
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method,
                    url,
                    headers=headers,
                    json=json_data,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"❌ HTTP {response.status}: {url}")
                        raise Exception(f"HTTP {response.status}")
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"❌ Max retries for {url}: {str(e)}")
                raise
            
            delay = (2 ** attempt)
            logger.warning(f"⚠️ Retry {attempt + 1}/{max_retries} for {url} after {delay}s. Error: {str(e)}")
            await asyncio.sleep(delay)

# ============================================================================
# FIRESTORE OPERATIONS
# ============================================================================

def write_to_firestore(collection: str, document_id: str, data: Dict) -> bool:
    """
    Write data to Firestore.
    
    Args:
        collection: Firestore collection name
        document_id: Document ID
        data: Data to write
    
    Returns:
        True if successful
    """
    try:
        db = get_firestore_client()
        db.collection(collection).document(document_id).set(data, merge=True)
        logger.info(f"✅ Wrote to {collection}/{document_id}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to write to {collection}/{document_id}: {str(e)}")
        return False

def read_from_firestore(collection: str, document_id: str) -> Optional[Dict]:
    """
    Read data from Firestore.
    
    Args:
        collection: Firestore collection name
        document_id: Document ID
    
    Returns:
        Document data or None
    """
    try:
        db = get_firestore_client()
        doc = db.collection(collection).document(document_id).get()
        if doc.exists:
            logger.info(f"✅ Read from {collection}/{document_id}")
            return doc.to_dict()
        else:
            logger.warning(f"⚠️ Document not found: {collection}/{document_id}")
            return None
    except Exception as e:
        logger.error(f"❌ Failed to read from {collection}/{document_id}: {str(e)}")
        return None

def query_firestore(
    collection: str,
    field: Optional[str] = None,
    operator: Optional[str] = None,
    value: Optional[Any] = None,
    order_by: Optional[str] = None,
    direction: Optional[str] = None,
    limit: int = 100
) -> List[Dict]:
    """
    Query Firestore collection.
    
    Args:
        collection: Collection name
        field: Field to filter on
        operator: Filter operator (==, <, >, <=, >=, !=)
        value: Filter value
        order_by: Field to order by
        direction: 'asc' or 'desc'
        limit: Max documents to return
    
    Returns:
        List of matching documents
    """
    try:
        db = get_firestore_client()
        query = db.collection(collection)
        
        if field and operator and value is not None:
            if operator == "==":
                query = query.where(field, "==", value)
            elif operator == "<":
                query = query.where(field, "<", value)
            elif operator == ">":
                query = query.where(field, ">", value)
            elif operator == "<=":
                query = query.where(field, "<=", value)
            elif operator == ">=":
                query = query.where(field, ">=", value)
            elif operator == "!=":
                query = query.where(field, "!=", value)
        
        if order_by:
            direction_enum = firestore.Query.DESCENDING if direction == "desc" else firestore.Query.ASCENDING
            query = query.order_by(order_by, direction=direction_enum)
        
        query = query.limit(limit)
        docs = query.stream()
        results = [doc.to_dict() for doc in docs]
        
        logger.info(f"✅ Query {collection}: {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"❌ Query failed for {collection}: {str(e)}")
        return []

def delete_from_firestore(collection: str, document_id: str) -> bool:
    """Delete document from Firestore."""
    try:
        db = get_firestore_client()
        db.collection(collection).document(document_id).delete()
        logger.info(f"✅ Deleted {collection}/{document_id}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to delete {collection}/{document_id}: {str(e)}")
        return False

# ============================================================================
# TIME UTILITIES
# ============================================================================

def get_utc_now() -> datetime:
    """Get current time in UTC."""
    return datetime.now(pytz.UTC)

def get_utc_timestamp() -> str:
    """Get current UTC timestamp as ISO 8601 string."""
    return get_utc_now().isoformat()

def utc_to_iso(dt: datetime) -> str:
    """Convert datetime to ISO 8601 string."""
    return dt.isoformat()

def iso_to_datetime(iso_str: str) -> datetime:
    """Convert ISO 8601 string to datetime."""
    return datetime.fromisoformat(iso_str.replace('Z', '+00:00'))

def hours_ago(hours: int) -> datetime:
    """Get datetime from N hours ago."""
    return get_utc_now() - timedelta(hours=hours)

def minutes_ago(minutes: int) -> datetime:
    """Get datetime from N minutes ago."""
    return get_utc_now() - timedelta(minutes=minutes)

def seconds_ago(seconds: int) -> datetime:
    """Get datetime from N seconds ago."""
    return get_utc_now() - timedelta(seconds=seconds)

# ============================================================================
# LOGGING UTILITIES
# ============================================================================

def log_info(message: str, **kwargs):
    """Log info message with extra data."""
    logger.info(f"{message} | {json.dumps(kwargs)}" if kwargs else message)

def log_warning(message: str, **kwargs):
    """Log warning message with extra data."""
    logger.warning(f"{message} | {json.dumps(kwargs)}" if kwargs else message)

def log_error(message: str, error: Optional[Exception] = None, **kwargs):
    """Log error message with exception."""
    error_str = str(error) if error else "Unknown error"
    logger.error(f"{message} | Error: {error_str} | {json.dumps(kwargs)}" if kwargs else f"{message} | Error: {error_str}")

def log_debug(message: str, **kwargs):
    """Log debug message with extra data."""
    logger.debug(f"{message} | {json.dumps(kwargs)}" if kwargs else message)

# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_positive_float(value: Any, name: str) -> float:
    """Validate that value is a positive float."""
    try:
        float_val = float(value)
        if float_val <= 0:
            raise ValueError(f"{name} must be positive")
        return float_val
    except Exception as e:
        logger.error(f"❌ Validation failed for {name}: {str(e)}")
        raise

def validate_positive_int(value: Any, name: str) -> int:
    """Validate that value is a positive integer."""
    try:
        int_val = int(value)
        if int_val <= 0:
            raise ValueError(f"{name} must be positive")
        return int_val
    except Exception as e:
        logger.error(f"❌ Validation failed for {name}: {str(e)}")
        raise

def validate_percentage(value: Any, name: str) -> float:
    """Validate that value is between 0 and 100."""
    try:
        float_val = float(value)
        if float_val < 0 or float_val > 100:
            raise ValueError(f"{name} must be between 0 and 100")
        return float_val
    except Exception as e:
        logger.error(f"❌ Validation failed for {name}: {str(e)}")
        raise

def validate_string(value: Any, name: str, min_length: int = 1) -> str:
    """Validate that value is a non-empty string."""
    if not isinstance(value, str) or len(value) < min_length:
        raise ValueError(f"{name} must be a string with at least {min_length} characters")
    return value

# ============================================================================
# CURRENCY & MATH UTILITIES
# ============================================================================

def round_to_satoshi(value: float) -> float:
    """Round to 8 decimal places (satoshi precision)."""
    return round(value, 8)

def round_to_cents(value: float) -> float:
    """Round to 2 decimal places (USD cents)."""
    return round(value, 2)

def calculate_pnl_percent(entry_price: float, exit_price: float) -> float:
    """Calculate P&L percentage."""
    if entry_price == 0:
        return 0
    return ((exit_price - entry_price) / entry_price) * 100

def calculate_atr(high: List[float], low: List[float], close: List[float], period: int = 14) -> float:
    """
    Calculate Average True Range.
    
    Args:
        high: List of high prices
        low: List of low prices
        close: List of close prices
        period: ATR period (default 14)
    
    Returns:
        ATR value
    """
    if len(high) < period or len(low) < period or len(close) < period:
        return 0
    
    tr_list = []
    for i in range(1, len(close)):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        tr_list.append(tr)
    
    return sum(tr_list[-period:]) / period if tr_list else 0

# ============================================================================
# CONFIG UTILITIES
# ============================================================================

def load_strategy_config(strategy_name: str) -> Dict:
    """Load strategy configuration from Firestore."""
    try:
        db = get_firestore_client()
        config_docs = db.collection("config").where("strategy", "==", strategy_name).stream()
        config = {}
        for doc in config_docs:
            config[doc.get("parameter_name")] = doc.get("parameter_value")
        
        logger.info(f"✅ Loaded config for {strategy_name}")
        return config
    except Exception as e:
        logger.error(f"❌ Failed to load config for {strategy_name}: {str(e)}")
        return {}

def update_strategy_config(strategy_name: str, parameter_name: str, parameter_value: Any) -> bool:
    """Update strategy parameter in Firestore."""
    try:
        db = get_firestore_client()
        doc_id = f"{strategy_name}_{parameter_name}"
        db.collection("config").document(doc_id).set({
            "strategy": strategy_name,
            "parameter_name": parameter_name,
            "parameter_value": parameter_value,
            "effective_from": get_utc_timestamp()
        }, merge=True)
        
        logger.info(f"✅ Updated {strategy_name}/{parameter_name}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to update config: {str(e)}")
        return False

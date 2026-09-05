import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import random
import numpy as np

from src.helpers import (
    query_firestore,
    write_to_firestore,
    get_utc_timestamp,
    get_firestore_client,
    log_info,
    log_error,
    load_strategy_config,
    update_strategy_config
)
from src.validation import ValidationEngine
from src.llm_strategist import LLMStrategist

logger = logging.getLogger(__name__)

# ============================================================================
# EVOLUTION ENGINE (Genetic Algorithm)
# ============================================================================

class EvolutionEngine:
    """
    Autonomous parameter evolution via genetic algorithm.
    
    Philosophy: "Build Once, Evolve Forever"
    - All features built upfront
    - System evolves autonomously via ICIR-driven mutation
    - No manual tuning after launch
    - Only monitoring and emergency intervention
    
    Evolution Process:
    1. Evaluate: Calculate ICIR on recent trades
    2. Validate: Run combined ICIR + Walk-Forward
    3. Mutate: Adjust parameters based on failures
    4. Select: Keep best performing variant
    5. Track: Record evolution history
    
    Parameters (Approach B - Momentum):
    - Z-score (entry threshold): 1.0 - 2.5
    - Take Profit %: 2% - 8%
    - Stop Loss %: 1% - 5%
    - Hold Time: 60 - 360 minutes
    
    Parameters (Approach C - Swing):
    - RSI Entry: 30 - 70
    - MA Period: 20 - 100
    - Volume Spike: 1.0 - 3.0x
    - Hold Time: 240 - 1440 minutes
    """
    
    # Evolution parameters
    POPULATION_SIZE = 10
    ELITE_SIZE = 2
    MUTATION_RATE = 0.3
    MUTATION_MAGNITUDE = 0.1  # 10% adjustment
    MAX_GENERATIONS = 100
    
    # Parameter bounds
    APPROACH_B_BOUNDS = {
        "z_score": (1.0, 2.5),
        "take_profit_percent": (0.02, 0.08),
        "stop_loss_percent": (0.01, 0.05),
        "hold_time_minutes": (60, 360)
    }
    
    APPROACH_C_BOUNDS = {
        "rsi_entry": (30, 70),
        "ma_period": (20, 100),
        "volume_spike": (1.0, 3.0),
        "hold_time_minutes": (240, 1440)
    }
    
    def __init__(self):
        self.db = get_firestore_client()
        self.validation_engine = ValidationEngine()
        self.llm_strategist = LLMStrategist()
    
    async def run_evolution_cycle(self) -> Dict:
        """
        Run a full evolution cycle.
        
        Steps:
        1. Load recent trades
        2. Calculate ICIR
        3. Generate parameter variations
        4. Validate each variant
        5. Select best
        6. Update config
        7. Track evolution
        
        Returns:
            Evolution report
        """
        try:
            logger.info("🧬 Starting evolution cycle...")
            
            # Step 1: Load recent trades
            recent_trades = await self._load_recent_trades(limit=50)
            if len(recent_trades) < 10:
                logger.warning(f"⚠️ Insufficient trades for evolution: {len(recent_trades)}")
                return {
                    "status": "insufficient_data",
                    "trade_count": len(recent_trades)
                }
            
            # Step 2: Calculate current ICIR
            current_icir = self._calculate_current_icir(recent_trades)
            logger.info(f"📊 Current ICIR: {current_icir:.4f}")
            
            # Step 3: Generate variants
            variants = await self._generate_population()
            logger.info(f"🔀 Generated {len(variants)} variants")
            
            # Step 4: Validate each variant
            validated_variants = []
            for variant in variants:
                try:
                    validation_result = await self.validation_engine.validate_strategy_variant(
                        variant["id"],
                        variant["params"]
                    )
                    
                    if validation_result.get("passes_validation"):
                        variant["validation"] = validation_result
                        variant["fitness"] = validation_result.get("icir", {}).get("out_of_sample_avg", 0)
                        validated_variants.append(variant)
                except Exception as e:
                    logger.warning(f"⚠️ Validation failed for {variant['id']}: {str(e)}")
                    continue
            
            logger.info(f"✅ {len(validated_variants)} variants passed validation")
            
            if not validated_variants:
                logger.warning("⚠️ No variants passed validation")
                return {
                    "status": "no_valid_variants",
                    "variants_tested": len(variants),
                    "variants_passed": 0
                }
            
            # Step 5: Select best variant
            best_variant = max(validated_variants, key=lambda v: v.get("fitness", 0))
            best_fitness = best_variant.get("fitness", 0)
            
            logger.info(f"🏆 Best variant: {best_variant['id']} (ICIR: {best_fitness:.4f})")
            
            # Step 6: Update config if better than current
            update_applied = False
            if best_fitness > current_icir:
                logger.info(f"📈 New variant better than current ({best_fitness:.4f} > {current_icir:.4f})")
                await self._apply_variant_config(best_variant)
                update_applied = True
            else:
                logger.info(f"⚠️ No improvement ({best_fitness:.4f} <= {current_icir:.4f})")
            
            # Step 7: Track evolution
            evolution_record = {
                "timestamp": get_utc_timestamp(),
                "generation": await self._get_generation_count(),
                "current_icir": current_icir,
                "best_variant_id": best_variant["id"],
                "best_variant_icir": best_fitness,
                "variants_generated": len(variants),
                "variants_validated": len(validated_variants),
                "update_applied": update_applied,
                "improvement": best_fitness - current_icir
            }
            
            await self._save_evolution_record(evolution_record)
            
            # Step 8: LLM analysis
            try:
                llm_analysis = await self.llm_strategist.analyze_evolution(
                    current_icir,
                    best_fitness,
                    recent_trades
                )
                evolution_record["llm_analysis"] = llm_analysis
            except Exception as e:
                logger.warning(f"⚠️ LLM analysis failed: {str(e)}")
            
            logger.info("✅ Evolution cycle complete")
            
            return evolution_record
        except Exception as e:
            log_error("❌ Evolution cycle failed", e)
            raise
    
    # ========================================================================
    # TRADE DATA
    # ========================================================================
    
    async def _load_recent_trades(self, limit: int = 50) -> List[Dict]:
        """Load recent closed trades."""
        try:
            trades = query_firestore(
                "trade_history",
                order_by="timestamp",
                direction="desc",
                limit=limit
            )
            
            logger.info(f"✅ Loaded {len(trades)} recent trades")
            return trades
        except Exception as e:
            log_error("❌ Failed to load trades", e)
            return []
    
    def _calculate_current_icir(self, trades: List[Dict]) -> float:
        """Calculate ICIR from recent trades."""
        try:
            if len(trades) < 10:
                return 0
            
            # Use validation engine to calculate
            icir = self.validation_engine._calculate_icir(trades)
            return icir
        except Exception as e:
            logger.error(f"❌ ICIR calculation failed: {str(e)}")
            return 0
    
    # ========================================================================
    # VARIANT GENERATION (Population)
    # ========================================================================
    
    async def _generate_population(self, size: int = None) -> List[Dict]:
        """
        Generate population of parameter variants.
        
        Includes:
        - Current best variant (elite)
        - Mutations of best variant
        - Random variants
        """
        try:
            if size is None:
                size = self.POPULATION_SIZE
            
            variants = []
            
            # Load current best parameters
            current_best = await self._load_best_parameters()
            
            # Add elite (current best)
            variants.append({
                "id": f"elite_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "params": current_best,
                "source": "elite"
            })
            
            # Add mutations of elite
            mutations_count = size // 2
            for i in range(mutations_count):
                mutated = self._mutate_parameters(current_best)
                variants.append({
                    "id": f"mutant_{i}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    "params": mutated,
                    "source": "mutation"
                })
            
            # Add random variants
            random_count = size - len(variants)
            for i in range(random_count):
                random_params = self._generate_random_parameters()
                variants.append({
                    "id": f"random_{i}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    "params": random_params,
                    "source": "random"
                })
            
            logger.info(f"✅ Generated {len(variants)} variants ({len(variants) - 1} mutations + random)")
            return variants
        except Exception as e:
            log_error("❌ Population generation failed", e)
            return []
    
    async def _load_best_parameters(self) -> Dict:
        """Load current best performing parameters."""
        try:
            # Load from config or use defaults
            params = {
                "approach_b": load_strategy_config("approach_b"),
                "approach_c": load_strategy_config("approach_c")
            }
            
            # Use defaults if empty
            if not params["approach_b"]:
                params["approach_b"] = self._get_default_parameters("approach_b")
            if not params["approach_c"]:
                params["approach_c"] = self._get_default_parameters("approach_c")
            
            return params
        except Exception as e:
            logger.error(f"❌ Failed to load best parameters: {str(e)}")
            return {
                "approach_b": self._get_default_parameters("approach_b"),
                "approach_c": self._get_default_parameters("approach_c")
            }
    
    def _mutate_parameters(self, params: Dict) -> Dict:
        """
        Mutate parameters (small random adjustments).
        
        Mutation strategy:
        - Random parameter selected
        - Adjusted by ±10% (MUTATION_MAGNITUDE)
        - Kept within bounds
        """
        try:
            mutated = {}
            
            for strategy, params_dict in params.items():
                mutated[strategy] = {}
                bounds = self.APPROACH_B_BOUNDS if strategy == "approach_b" else self.APPROACH_C_BOUNDS
                
                for param_name, param_value in params_dict.items():
                    if param_name in bounds:
                        # Mutate with probability
                        if random.random() < self.MUTATION_RATE:
                            # Random adjustment
                            adjustment = random.uniform(
                                -self.MUTATION_MAGNITUDE,
                                self.MUTATION_MAGNITUDE
                            )
                            mutated_value = param_value * (1 + adjustment)
                            
                            # Apply bounds
                            lower, upper = bounds[param_name]
                            mutated_value = max(lower, min(upper, mutated_value))
                            mutated[strategy][param_name] = mutated_value
                        else:
                            mutated[strategy][param_name] = param_value
                    else:
                        mutated[strategy][param_name] = param_value
            
            return mutated
        except Exception as e:
            logger.error(f"❌ Parameter mutation failed: {str(e)}")
            return params
    
    def _generate_random_parameters(self) -> Dict:
        """Generate completely random parameters within bounds."""
        try:
            return {
                "approach_b": {
                    "z_score": random.uniform(*self.APPROACH_B_BOUNDS["z_score"]),
                    "take_profit_percent": random.uniform(*self.APPROACH_B_BOUNDS["take_profit_percent"]),
                    "stop_loss_percent": random.uniform(*self.APPROACH_B_BOUNDS["stop_loss_percent"]),
                    "hold_time_minutes": random.randint(*self.APPROACH_B_BOUNDS["hold_time_minutes"])
                },
                "approach_c": {
                    "rsi_entry": random.randint(*self.APPROACH_C_BOUNDS["rsi_entry"]),
                    "ma_period": random.randint(*self.APPROACH_C_BOUNDS["ma_period"]),
                    "volume_spike": random.uniform(*self.APPROACH_C_BOUNDS["volume_spike"]),
                    "hold_time_minutes": random.randint(*self.APPROACH_C_BOUNDS["hold_time_minutes"])
                }
            }
        except Exception as e:
            logger.error(f"❌ Random parameter generation failed: {str(e)}")
            return self._get_default_parameters("both")
    
    def _get_default_parameters(self, strategy: str = "both") -> Dict:
        """Get default parameters for initialization."""
        defaults = {
            "approach_b": {
                "z_score": 1.6,
                "take_profit_percent": 0.048,
                "stop_loss_percent": 0.032,
                "hold_time_minutes": 180
            },
            "approach_c": {
                "rsi_entry": 60,
                "ma_period": 50,
                "volume_spike": 1.5,
                "hold_time_minutes": 480
            }
        }
        
        if strategy == "both":
            return defaults
        elif strategy in defaults:
            return defaults[strategy]
        else:
            return {}
    
    # ========================================================================
    # VARIANT APPLICATION
    # ========================================================================
    
    async def _apply_variant_config(self, variant: Dict) -> bool:
        """Apply best variant's configuration to system."""
        try:
            variant_id = variant.get("id")
            params = variant.get("params", {})
            
            logger.info(f"📝 Applying variant {variant_id} configuration...")
            
            for strategy, param_dict in params.items():
                for param_name, param_value in param_dict.items():
                    success = update_strategy_config(strategy, param_name, param_value)
                    if success:
                        logger.info(f"✅ Updated {strategy}.{param_name} = {param_value}")
            
            logger.info(f"✅ Applied variant {variant_id}")
            return True
        except Exception as e:
            log_error("❌ Failed to apply variant", e)
            return False
    
    # ========================================================================
    # EVOLUTION TRACKING
    # ========================================================================
    
    async def _get_generation_count(self) -> int:
        """Get current generation number."""
        try:
            records = query_firestore(
                "evolution_history",
                order_by="generation",
                direction="desc",
                limit=1
            )
            
            if records:
                return records[0].get("generation", 0) + 1
            else:
                return 1
        except Exception as e:
            logger.error(f"❌ Failed to get generation count: {str(e)}")
            return 1
    
    async def _save_evolution_record(self, record: Dict) -> bool:
        """Save evolution cycle record to Firestore."""
        try:
            generation = record.get("generation", 0)
            doc_id = f"gen_{generation}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            write_to_firestore("evolution_history", doc_id, record)
            
            logger.info(f"✅ Saved evolution record: {doc_id}")
            return True
        except Exception as e:
            log_error("❌ Failed to save evolution record", e)
            return False
    
    def get_evolution_summary(self) -> str:
        """Generate human-readable evolution summary."""
        try:
            records = query_firestore(
                "evolution_history",
                order_by="timestamp",
                direction="desc",
                limit=10
            )
            
            if not records:
                return "No evolution history"
            
            latest = records[0]
            
            summary = f"""
            🧬 EVOLUTION SUMMARY
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            Generation: {latest.get('generation', 0)}
            Current ICIR: {latest.get('current_icir', 0):.4f}
            Best Variant ICIR: {latest.get('best_variant_icir', 0):.4f}
            Improvement: {latest.get('improvement', 0):+.4f}
            Update Applied: {'✅' if latest.get('update_applied') else '❌'}
            
            Variants:
            - Generated: {latest.get('variants_generated', 0)}
            - Validated: {latest.get('variants_validated', 0)}
            
            Best Variant: {latest.get('best_variant_id', 'N/A')}
            """
            
            return summary.strip()
        except Exception as e:
            logger.error(f"❌ Failed to generate summary: {str(e)}")
            return "Error generating summary"

# ============================================================================
# MAIN FUNCTION
# ============================================================================

async def evolution_engine():
    """Main entry point for evolution cycle."""
    try:
        engine = EvolutionEngine()
        report = await engine.run_evolution_cycle()
        return report
    except Exception as e:
        log_error("❌ Evolution engine failed", e)
        raise

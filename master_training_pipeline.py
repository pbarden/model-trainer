#!/usr/bin/env python3
"""
Model Tea - Master Training Pipeline
Copyright © ChaiQ LLC

Complete training orchestration system that:
1. Identifies incomplete individual novels and combined models
2. Trains all missing models using existing training scripts
3. Integrates with optimized Model Tea framework for lifecycle management
4. Generates quality and performance reports after each model
5. Processes episodic memories for all trained models
6. Delivers complete training pipeline execution
"""

import os
import sys
import json
import time
import logging
import argparse
import warnings
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from model_tea.core.models import ModelRegistry, ModelManager
from model_tea_utils import ModelTeaConfig, FileSystemUtils, ErrorHandling, validate_system_setup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MasterPipelineConfig:
    """Configuration for complete training pipeline"""
    parallel_individual_training: bool = True
    max_parallel_novels: int = 4
    force_retrain_individual: bool = False
    force_retrain_combined: bool = False

    run_quality_evaluation: bool = True
    run_performance_testing: bool = True
    generate_detailed_reports: bool = True

    process_episodic_memories: bool = True

    model_storage_path: str = "model_storage"
    registry_path: str = "model_registry"

    stop_on_error: bool = False
    timeout_per_novel: int = 3600 
    timeout_per_combined: int = 7200 


class CompletePipelineOrchestrator:
    """Complete training pipeline orchestrator with Model Tea integration"""

    def __init__(self, config: MasterPipelineConfig = None):
        self.config = config or MasterPipelineConfig()

        self.registry = ModelRegistry(self.config.registry_path)
        self.manager = ModelManager(self.registry, self.config.model_storage_path)

        self.model_mapping = self._load_model_mapping()
        validate_system_setup()

        self.execution_stats = {
            'novels_trained': 0,
            'combined_models_trained': 0,
            'reports_generated': 0,
            'episodic_memories_processed': 0,
            'total_execution_time': 0,
            'errors': []
        }

    def _load_model_mapping(self) -> Dict[str, Any]:
        """Load model mapping configuration"""
        mapping_file = Path("model_mapping.json")
        if not mapping_file.exists():
            raise FileNotFoundError("model_mapping.json not found")

        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Failed to load model_mapping.json: {e}")

    def get_all_novels(self) -> Set[str]:
        """Get complete set of all novels across all models"""
        all_novels = set()

        if "models" in self.model_mapping:
            for model_key, model_info in self.model_mapping["models"].items():
                for novel in model_info.get("novels", []):
                    all_novels.add(novel["directory_name"])

        return all_novels

    def get_all_combined_models(self) -> List[str]:
        """Get list of all combined models to train"""
        if "models" not in self.model_mapping:
            return []
        return list(self.model_mapping["models"].keys())

    def identify_incomplete_novels(self) -> List[str]:
        """Identify novels that haven't been trained or need retraining"""
        all_novels = self.get_all_novels()
        incomplete_novels = []

        for novel_name in all_novels:
            # Check if novel exists in registry
            model_id = f"novel_{novel_name}"
            existing_model = self.registry.get_model(model_id)

            # Check traditional file location as backup
            traditional_path = Path("iterative_models") / novel_name / "final"

            if (not existing_model and not traditional_path.exists()) or self.config.force_retrain_individual:
                incomplete_novels.append(novel_name)
                logger.info(f"Novel needs training: {novel_name}")
            else:
                logger.info(f"Novel already trained: {novel_name}")

        return incomplete_novels

    def identify_incomplete_combined_models(self) -> List[str]:
        """Identify combined models that haven't been trained"""
        all_combined = self.get_all_combined_models()
        incomplete_combined = []

        for model_key in all_combined:
            # Check if combined model exists in registry
            existing_model = self.registry.get_model(model_key)

            # Check traditional file location as backup
            traditional_path = Path("iterative_models") / model_key / "final"

            if (not existing_model and not traditional_path.exists()) or self.config.force_retrain_combined:
                incomplete_combined.append(model_key)
                logger.info(f"Combined model needs training: {model_key}")
            else:
                logger.info(f"Combined model already trained: {model_key}")

        return incomplete_combined

    def train_single_novel(self, novel_name: str) -> Dict[str, Any]:
        """Train a single novel using iterative_novel_trainer.py"""
        result = {
            'novel_name': novel_name,
            'status': 'running',
            'start_time': time.time()
        }

        try:
            logger.info(f"Training individual novel: {novel_name}")

            # Validate novel directory exists
            novel_path = Path("novels") / novel_name
            if not novel_path.exists():
                result['status'] = 'failed'
                result['error'] = f"Novel directory not found: {novel_path}"
                return result

            # Run iterative novel trainer
            cmd = ["python", "iterative_novel_trainer.py", "--novel", novel_name]

            process_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_per_novel
            )

            if process_result.returncode == 0:
                # Training successful - register with Model Tea
                model_path = Path("iterative_models") / novel_name / "final"

                if model_path.exists():
                    # Load training results if available
                    results_file = Path("iterative_models") / novel_name / "training_results.json"
                    training_metrics = {}

                    if results_file.exists():
                        try:
                            with open(results_file, 'r') as f:
                                training_data = json.load(f)
                                training_metrics = {
                                    'final_quality': training_data.get('final_quality_score', 0),
                                    'training_iterations': training_data.get('iterations_completed', 0),
                                    'total_training_time': training_data.get('total_time_minutes', 0)
                                }
                        except Exception as e:
                            logger.warning(f"Could not load training results for {novel_name}: {e}")

                    # Create model object for registry
                    novel_model = {
                        "novel_name": novel_name,
                        "model_path": str(model_path),
                        "training_completed": True
                    }

                    # Save to Model Tea registry
                    saved_model = self.manager.save_model(
                        model=novel_model,
                        model_id=f"novel_{novel_name}",
                        status="production",
                        model_type="individual_novel",
                        novel_source=novel_name,
                        **training_metrics
                    )

                    result['status'] = 'completed'
                    result['model_version'] = saved_model.metadata.version
                    logger.info(f"Novel training completed and registered: {novel_name}")
                else:
                    result['status'] = 'failed'
                    result['error'] = "Model directory not created after training"
            else:
                result['status'] = 'failed'
                result['error'] = f"Training process failed: {process_result.stderr}"

        except subprocess.TimeoutExpired:
            result['status'] = 'timeout'
            result['error'] = f"Training timeout ({self.config.timeout_per_novel}s)"
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
        finally:
            result['duration'] = time.time() - result['start_time']

        return result

    def train_novels_batch(self, novel_names: List[str]) -> Dict[str, Any]:
        """Train multiple novels in parallel or sequentially"""
        if not novel_names:
            return {'novels_trained': 0, 'results': {}}

        results = {}
        novels_trained = 0

        if self.config.parallel_individual_training and len(novel_names) > 1:
            # Parallel training
            max_workers = min(self.config.max_parallel_novels, len(novel_names))
            logger.info(f"Training {len(novel_names)} novels with {max_workers} parallel workers")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_novel = {
                    executor.submit(self.train_single_novel, novel): novel
                    for novel in novel_names
                }

                for future in as_completed(future_to_novel):
                    novel_name = future_to_novel[future]
                    try:
                        result = future.result()
                        results[novel_name] = result

                        if result['status'] == 'completed':
                            novels_trained += 1
                            self.execution_stats['novels_trained'] += 1

                    except Exception as e:
                        results[novel_name] = {'status': 'error', 'error': str(e)}
                        self.execution_stats['errors'].append(f"Novel {novel_name}: {e}")
        else:
            # Sequential training
            logger.info(f"Training {len(novel_names)} novels sequentially")
            for novel_name in novel_names:
                result = self.train_single_novel(novel_name)
                results[novel_name] = result

                if result['status'] == 'completed':
                    novels_trained += 1
                    self.execution_stats['novels_trained'] += 1

        return {'novels_trained': novels_trained, 'results': results}

    def train_combined_model(self, model_key: str) -> Dict[str, Any]:
        """Train a combined model using combined_model_trainer.py"""
        result = {
            'model_key': model_key,
            'status': 'running',
            'start_time': time.time()
        }

        try:
            logger.info(f"Training combined model: {model_key}")

            # Run combined model trainer
            cmd = ["python", "combined_model_trainer.py", "--model", model_key]
            if self.config.force_retrain_combined:
                cmd.append("--force")

            process_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_per_combined
            )

            if process_result.returncode == 0:
                # Training successful - register with Model Tea
                model_path = Path("iterative_models") / model_key / "final"

                if model_path.exists():
                    # Load training results
                    results_file = Path("iterative_models") / model_key / "training_results.json"
                    training_metrics = {}

                    if results_file.exists():
                        try:
                            with open(results_file, 'r') as f:
                                training_data = json.load(f)
                                training_metrics = {
                                    'final_quality': training_data.get('final_quality', 0),
                                    'training_time': training_data.get('training_time_minutes', 0),
                                    'novels_included': len(training_data.get('novels_included', []))
                                }
                        except Exception as e:
                            logger.warning(f"Could not load training results for {model_key}: {e}")

                    # Get novels for this model
                    novels = []
                    if model_key in self.model_mapping.get("models", {}):
                        novels = [n["directory_name"] for n in self.model_mapping["models"][model_key]["novels"]]

                    # Create combined model object
                    combined_model = {
                        "model_key": model_key,
                        "model_path": str(model_path),
                        "component_novels": novels,
                        "training_completed": True
                    }

                    # Save to Model Tea registry
                    saved_model = self.manager.save_model(
                        model=combined_model,
                        model_id=model_key,
                        status="production",
                        model_type="combined_model",
                        component_novels=len(novels),
                        **training_metrics
                    )

                    result['status'] = 'completed'
                    result['model_version'] = saved_model.metadata.version
                    self.execution_stats['combined_models_trained'] += 1
                    logger.info(f"Combined model training completed and registered: {model_key}")
                else:
                    result['status'] = 'failed'
                    result['error'] = "Model directory not created after training"
            else:
                result['status'] = 'failed'
                result['error'] = f"Training process failed: {process_result.stderr}"

        except subprocess.TimeoutExpired:
            result['status'] = 'timeout'
            result['error'] = f"Training timeout ({self.config.timeout_per_combined}s)"
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
        finally:
            result['duration'] = time.time() - result['start_time']

        return result

    def generate_quality_report(self, model_id: str) -> Dict[str, Any]:
        """Generate quality and performance report for a trained model"""
        if not self.config.run_quality_evaluation:
            return {'status': 'skipped'}

        try:
            logger.info(f"Generating quality report for: {model_id}")

            # Run post-training evaluator
            cmd = ["python", "post_training_evaluator.py", "--model", model_id]

            process_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for evaluation
            )

            if process_result.returncode == 0:
                self.execution_stats['reports_generated'] += 1
                logger.info(f"Quality report generated for: {model_id}")
                return {'status': 'completed', 'output': process_result.stdout}
            else:
                logger.warning(f"Quality report generation failed for {model_id}")
                return {'status': 'failed', 'error': process_result.stderr}

        except Exception as e:
            logger.error(f"Error generating quality report for {model_id}: {e}")
            return {'status': 'error', 'error': str(e)}

    def process_episodic_memories(self, model_id: str) -> Dict[str, Any]:
        """Process episodic memories for a trained model"""
        if not self.config.process_episodic_memories:
            return {'status': 'skipped'}

        try:
            logger.info(f"Processing episodic memories for: {model_id}")

            # Run episodic memory system
            cmd = ["python", "episodic_memory_system.py", "--model", model_id]

            process_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout for memory processing
            )

            if process_result.returncode == 0:
                self.execution_stats['episodic_memories_processed'] += 1
                logger.info(f"Episodic memories processed for: {model_id}")
                return {'status': 'completed', 'output': process_result.stdout}
            else:
                logger.warning(f"Episodic memory processing failed for {model_id}")
                return {'status': 'failed', 'error': process_result.stderr}

        except Exception as e:
            logger.error(f"Error processing episodic memories for {model_id}: {e}")
            return {'status': 'error', 'error': str(e)}

    def execute_complete_pipeline(self) -> Dict[str, Any]:
        """Execute the complete training pipeline"""
        pipeline_start_time = time.time()
        logger.info("Starting complete training pipeline execution")

        pipeline_result = {
            'status': 'running',
            'start_time': pipeline_start_time,
            'stages': {}
        }

        try:
            # Stage 1: Identify incomplete work
            logger.info("=== Stage 1: Identifying incomplete training work ===")
            incomplete_novels = self.identify_incomplete_novels()
            incomplete_combined = self.identify_incomplete_combined_models()

            pipeline_result['stages']['identification'] = {
                'incomplete_novels': incomplete_novels,
                'incomplete_combined': incomplete_combined,
                'total_novels_needed': len(incomplete_novels),
                'total_combined_needed': len(incomplete_combined)
            }

            logger.info(f"Found {len(incomplete_novels)} novels and {len(incomplete_combined)} combined models to train")

            # Stage 2: Train individual novels
            if incomplete_novels:
                logger.info("=== Stage 2: Training individual novels ===")
                novel_results = self.train_novels_batch(incomplete_novels)
                pipeline_result['stages']['individual_training'] = novel_results

                # Generate reports for completed novels
                for novel_name, result in novel_results['results'].items():
                    if result['status'] == 'completed':
                        model_id = f"novel_{novel_name}"
                        self.generate_quality_report(model_id)
                        self.process_episodic_memories(model_id)
            else:
                logger.info("=== Stage 2: All individual novels already trained ===")
                pipeline_result['stages']['individual_training'] = {'novels_trained': 0, 'results': {}}

            # Stage 3: Train combined models
            if incomplete_combined:
                logger.info("=== Stage 3: Training combined models ===")
                combined_results = {}

                for model_key in incomplete_combined:
                    result = self.train_combined_model(model_key)
                    combined_results[model_key] = result

                    # Generate reports for completed combined models
                    if result['status'] == 'completed':
                        self.generate_quality_report(model_key)
                        self.process_episodic_memories(model_key)

                pipeline_result['stages']['combined_training'] = {
                    'models_trained': sum(1 for r in combined_results.values() if r['status'] == 'completed'),
                    'results': combined_results
                }
            else:
                logger.info("=== Stage 3: All combined models already trained ===")
                pipeline_result['stages']['combined_training'] = {'models_trained': 0, 'results': {}}

            # Final statistics
            total_time = time.time() - pipeline_start_time
            self.execution_stats['total_execution_time'] = total_time

            pipeline_result['status'] = 'completed'
            pipeline_result['execution_stats'] = self.execution_stats
            pipeline_result['total_duration'] = total_time

            logger.info("=== Pipeline Execution Complete ===")
            logger.info(f"Novels trained: {self.execution_stats['novels_trained']}")
            logger.info(f"Combined models trained: {self.execution_stats['combined_models_trained']}")
            logger.info(f"Reports generated: {self.execution_stats['reports_generated']}")
            logger.info(f"Episodic memories processed: {self.execution_stats['episodic_memories_processed']}")
            logger.info(f"Total execution time: {total_time/60:.2f} minutes")

        except Exception as e:
            pipeline_result['status'] = 'failed'
            pipeline_result['error'] = str(e)
            self.execution_stats['errors'].append(str(e))
            logger.error(f"Pipeline execution failed: {e}")

        return pipeline_result

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status across all models"""
        all_novels = self.get_all_novels()
        all_combined = self.get_all_combined_models()

        novel_status = {}
        for novel in all_novels:
            model_id = f"novel_{novel}"
            existing_model = self.registry.get_model(model_id)
            novel_status[novel] = {
                'trained': existing_model is not None,
                'model_version': existing_model.metadata.version if existing_model else None
            }

        combined_status = {}
        for model_key in all_combined:
            existing_model = self.registry.get_model(model_key)
            combined_status[model_key] = {
                'trained': existing_model is not None,
                'model_version': existing_model.metadata.version if existing_model else None
            }

        return {
            'novels': novel_status,
            'combined_models': combined_status,
            'summary': {
                'novels_trained': sum(1 for status in novel_status.values() if status['trained']),
                'novels_total': len(all_novels),
                'combined_trained': sum(1 for status in combined_status.values() if status['trained']),
                'combined_total': len(all_combined)
            }
        }


def main():
    """Main entry point for complete training pipeline"""
    parser = argparse.ArgumentParser(description="Complete Model Tea Training Pipeline")
    parser.add_argument("--status", action="store_true", help="Show current training status")
    parser.add_argument("--force-individual", action="store_true", help="Force retrain all individual novels")
    parser.add_argument("--force-combined", action="store_true", help="Force retrain all combined models")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel individual training")
    parser.add_argument("--no-reports", action="store_true", help="Skip quality reports and performance testing")
    parser.add_argument("--no-memory", action="store_true", help="Skip episodic memory processing")

    args = parser.parse_args()

    # Create configuration
    config = MasterPipelineConfig(
        force_retrain_individual=args.force_individual,
        force_retrain_combined=args.force_combined,
        parallel_individual_training=args.parallel,
        run_quality_evaluation=not args.no_reports,
        run_performance_testing=not args.no_reports,
        process_episodic_memories=not args.no_memory
    )

    # Create orchestrator
    orchestrator = CompletePipelineOrchestrator(config)

    if args.status:
        status = orchestrator.get_training_status()
        logger.info("=== Training Status ===")
        logger.info(f"Novels: {status['summary']['novels_trained']}/{status['summary']['novels_total']} trained")
        logger.info(f"Combined models: {status['summary']['combined_trained']}/{status['summary']['combined_total']} trained")

        if status['summary']['novels_trained'] < status['summary']['novels_total']:
            untrained_novels = [k for k, v in status['novels'].items() if not v['trained']]
            logger.info(f"Untrained novels: {untrained_novels}")

        if status['summary']['combined_trained'] < status['summary']['combined_total']:
            untrained_combined = [k for k, v in status['combined_models'].items() if not v['trained']]
            logger.info(f"Untrained combined models: {untrained_combined}")

        return

    # Execute complete pipeline
    result = orchestrator.execute_complete_pipeline()

    if result['status'] == 'completed':
        logger.info("Training pipeline completed successfully")
    else:
        logger.error(f"Training pipeline failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
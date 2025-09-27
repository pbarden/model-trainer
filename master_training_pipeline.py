#!/usr/bin/env python3
"""
Model Tea - Master Training Pipeline
Copyright © ChaiQ LLC

Master orchestration script that coordinates the complete training pipeline:
1. Individual novel training (iterative_novel_trainer.py)
2. Combined model training (combined_model_trainer.py)
3. Relational memory mapping (relational_memory_mapper.py)
4. Validation and reporting

Provides complete automation for training combined models like vs_mintchip.
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
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import Model Tea utilities
from model_tea_utils import (
    ModelTeaConfig, FileSystemUtils, ErrorHandling, validate_system_setup
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PipelineStage(Enum):
    """Pipeline stages enumeration"""
    INDIVIDUAL_TRAINING = "individual_training"
    COMBINED_TRAINING = "combined_training"
    RELATIONAL_MAPPING = "relational_mapping"
    VALIDATION = "validation"
    REPORTING = "reporting"

@dataclass
class PipelineConfig:
    """Configuration for master training pipeline"""
    # Stage control
    run_individual_training: bool = True
    run_combined_training: bool = True
    run_relational_mapping: bool = True
    run_validation: bool = True
    generate_report: bool = True

    # Individual training settings
    force_retrain_individual: bool = False
    individual_training_parallel: bool = False
    max_parallel_novels: int = 3

    # Combined training settings
    force_retrain_combined: bool = False
    skip_if_exists: bool = True

    # Pipeline behavior
    stop_on_error: bool = False
    save_intermediate_results: bool = True
    verbose_logging: bool = True

    # Output settings
    results_directory: str = "pipeline_results"
    generate_summary_report: bool = True


class MasterTrainingPipeline:
    """
    Master orchestrator for the complete training pipeline
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.model_mapping = self._load_model_mapping()

        # Setup directories
        self.novels_dir = Path("novels")
        self.models_dir = Path("iterative_models")
        self.results_dir = Path(self.config.results_directory)

        # Create results directory
        FileSystemUtils.ensure_directory(self.results_dir)

        # Validate system setup
        validate_system_setup()

        # Pipeline state
        self.pipeline_results = {}
        self.start_time = None

    def _load_model_mapping(self) -> Dict[str, Any]:
        """Load model mapping configuration"""
        mapping_file = Path("model_mapping.json")
        if not mapping_file.exists():
            raise FileNotFoundError("model_mapping.json not found.")

        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Failed to load model_mapping.json: {e}")

    def get_available_models(self) -> List[str]:
        """Get list of available combined models"""
        if "models" not in self.model_mapping:
            return []
        return list(self.model_mapping["models"].keys())

    def get_novels_for_model(self, model_key: str) -> List[Dict[str, Any]]:
        """Get novel list for specified combined model"""
        if "models" not in self.model_mapping or model_key not in self.model_mapping["models"]:
            raise ValueError(f"Model '{model_key}' not found in mapping")
        return self.model_mapping["models"][model_key]["novels"]

    def check_individual_novels_status(self, model_key: str) -> Dict[str, Any]:
        """Check training status of individual novels for a model"""
        novels = self.get_novels_for_model(model_key)

        status = {
            "total_novels": len(novels),
            "trained_novels": 0,
            "untrained_novels": 0,
            "novel_status": {}
        }

        for novel_info in novels:
            directory_name = novel_info["directory_name"]
            model_path = self.models_dir / directory_name / "final"
            is_trained = model_path.exists()

            status["novel_status"][directory_name] = {
                "original_name": novel_info["original_name"],
                "trained": is_trained,
                "model_path": str(model_path) if is_trained else None
            }

            if is_trained:
                status["trained_novels"] += 1
            else:
                status["untrained_novels"] += 1

        return status

    def check_combined_model_status(self, model_key: str) -> Dict[str, Any]:
        """Check if combined model is already trained"""
        model_path = self.models_dir / model_key / "final"
        results_file = self.models_dir / model_key / "training_results.json"

        status = {
            "model_exists": model_path.exists(),
            "results_exist": results_file.exists(),
            "model_path": str(model_path) if model_path.exists() else None,
            "results_path": str(results_file) if results_file.exists() else None
        }

        if status["results_exist"]:
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    status["training_completed"] = True
                    status["training_time"] = results.get("training_time", 0)
                    status["final_quality"] = results.get("final_quality", 0)
            except Exception as e:
                logger.warning(f"Failed to read training results for {model_key}: {e}")
                status["training_completed"] = False

        return status

    def check_relational_mappings_status(self, model_key: str) -> Dict[str, Any]:
        """Check if relational mappings exist for a model"""
        mappings_file = Path("relational_memories") / f"{model_key}_relational_mappings.json"

        status = {
            "mappings_exist": mappings_file.exists(),
            "mappings_path": str(mappings_file) if mappings_file.exists() else None
        }

        if status["mappings_exist"]:
            try:
                with open(mappings_file, 'r', encoding='utf-8') as f:
                    mappings = json.load(f)
                    status["mapping_completed"] = True
                    status["novel_count"] = mappings.get("novel_count", 0)
                    status["cross_references"] = sum(len(refs) for refs in mappings.get("cross_references", {}).values())
            except Exception as e:
                logger.warning(f"Failed to read relational mappings for {model_key}: {e}")
                status["mapping_completed"] = False

        return status

    def _train_single_novel(self, novel_name: str) -> Dict[str, Any]:
        """Train a single novel - used for parallel execution"""
        result = {
            "novel_name": novel_name,
            "status": "running",
            "start_time": time.time()
        }

        try:
            # Validate novel exists before training
            novel_path = Path("novels") / novel_name
            if not novel_path.exists():
                result["status"] = "failed"
                result["error"] = f"Novel directory not found: {novel_path}"
                return result

            # Run individual novel trainer with specific novel name
            cmd = ["python", "iterative_novel_trainer.py", "--novel", novel_name]

            # Train the specific novel for this model
            process_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout per novel
            )

            if process_result.returncode == 0:
                result["status"] = "success"
                result["output"] = process_result.stdout
            else:
                result["status"] = "failed"
                result["error"] = process_result.stderr

        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
            result["error"] = "Training timeout (1 hour)"
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
        finally:
            result["duration"] = time.time() - result["start_time"]

        return result

    def _train_novels_parallel(self, novels_to_train: List[str]) -> Dict[str, Any]:
        """Train multiple novels in parallel"""
        if not novels_to_train:
            return {
                "novels_trained": 0,
                "training_results": {},
                "total_novels": 0
            }

        max_workers = min(self.config.max_parallel_novels, len(novels_to_train))
        max_workers = max(1, max_workers)  # Ensure at least 1 worker
        logger.info(f"Training {len(novels_to_train)} novels with {max_workers} parallel workers")

        training_results = {}
        novels_trained = 0

        # Thread-safe progress tracking
        progress_lock = threading.Lock()
        completed_count = [0]  # Use list for mutable reference

        def log_progress(novel_name: str, status: str):
            with progress_lock:
                completed_count[0] += 1
                logger.info(f"  [{completed_count[0]}/{len(novels_to_train)}] {novel_name}: {status}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all training jobs
            future_to_novel = {
                executor.submit(self._train_single_novel, novel_name): novel_name
                for novel_name in novels_to_train
            }

            # Process completed futures
            for future in as_completed(future_to_novel):
                novel_name = future_to_novel[future]
                try:
                    result = future.result()
                    training_results[novel_name] = result

                    if result["status"] == "success":
                        novels_trained += 1
                        log_progress(novel_name, "SUCCESS")
                    else:
                        log_progress(novel_name, f"FAILED: {result['status'].upper()}")

                        # Stop all remaining jobs if stop_on_error is enabled
                        if self.config.stop_on_error and result["status"] in ["failed", "error", "timeout"]:
                            logger.error(f"Stopping all training due to failure in {novel_name}")
                            # Cancel remaining futures
                            for remaining_future in future_to_novel:
                                if not remaining_future.done():
                                    remaining_future.cancel()
                            break

                except Exception as e:
                    training_results[novel_name] = {
                        "status": "error",
                        "error": f"Future execution failed: {str(e)}"
                    }
                    log_progress(novel_name, "ERROR")

        return {
            "novels_trained": novels_trained,
            "training_results": training_results,
            "total_novels": len(novels_to_train)
        }

    def _train_novels_sequential(self, novels_to_train: List[str]) -> Dict[str, Any]:
        """Train novels sequentially (original implementation)"""
        novels_trained = 0
        training_results = {}

        for i, novel_name in enumerate(novels_to_train):
            logger.info(f"\nTraining novel {i+1}/{len(novels_to_train)}: {novel_name}")

            try:
                # Validate novel exists before training
                novel_path = Path("novels") / novel_name
                if not novel_path.exists():
                    training_results[novel_name] = {
                        "status": "failed",
                        "error": f"Novel directory not found: {novel_path}"
                    }
                    logger.error(f"  Novel directory not found: {novel_path}")
                    if self.config.stop_on_error:
                        break
                    continue

                # Run individual novel trainer with specific novel name
                cmd = ["python", "iterative_novel_trainer.py", "--novel", novel_name]
                logger.info(f"Running: {' '.join(cmd)}")

                # Train the specific novel for this model
                process_result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout per novel
                )

                if process_result.returncode == 0:
                    novels_trained += 1
                    training_results[novel_name] = {"status": "success", "output": process_result.stdout}
                    logger.info(f"  Successfully trained: {novel_name}")
                else:
                    training_results[novel_name] = {"status": "failed", "error": process_result.stderr}
                    logger.error(f"  Failed to train: {novel_name}")
                    if self.config.stop_on_error:
                        break

            except subprocess.TimeoutExpired:
                training_results[novel_name] = {"status": "timeout", "error": "Training timeout (1 hour)"}
                logger.error(f"  Training timeout: {novel_name}")
                if self.config.stop_on_error:
                    break

        return {
            "novels_trained": novels_trained,
            "training_results": training_results,
            "total_novels": len(novels_to_train)
        }

    def stage_1_individual_training(self, model_key: str) -> Dict[str, Any]:
        """Stage 1: Train individual novels"""
        logger.info(f"=== Stage 1: Individual Novel Training for {model_key} ===")

        stage_start = time.time()
        result = {
            "stage": "individual_training",
            "model_key": model_key,
            "status": "running",
            "start_time": stage_start
        }

        try:
            # Check current status
            novel_status = self.check_individual_novels_status(model_key)
            result["initial_status"] = novel_status

            logger.info(f"Individual novels status: {novel_status['trained_novels']}/{novel_status['total_novels']} trained")

            if novel_status["untrained_novels"] == 0 and not self.config.force_retrain_individual:
                logger.info("All individual novels already trained. Skipping stage 1.")
                result["status"] = "skipped"
                result["reason"] = "all_novels_already_trained"
                return result

            # Get list of novels for this model
            novels = self.get_novels_for_model(model_key)
            novels_to_train = []

            # Validate that all novels exist in the filesystem
            novels_dir = Path("novels")
            for novel_info in novels:
                directory_name = novel_info["directory_name"]
                novel_path = novels_dir / directory_name

                if not novel_path.exists():
                    logger.error(f"Novel directory not found: {novel_path}")
                    logger.error(f"Expected from model_mapping.json: {directory_name}")
                    result["status"] = "error"
                    result["error"] = f"Novel directory not found: {directory_name}"
                    return result

                if not novel_status["novel_status"][directory_name]["trained"] or self.config.force_retrain_individual:
                    novels_to_train.append(directory_name)
                    logger.info(f"Will train: {directory_name}")
                else:
                    logger.info(f"Already trained: {directory_name}")

            if not novels_to_train:
                logger.info("No novels need training.")
                result["status"] = "completed"
                result["novels_trained"] = 0
                return result

            logger.info(f"Training {len(novels_to_train)} novels: {', '.join(novels_to_train)}")

            # Train novels using parallel or sequential execution
            if self.config.individual_training_parallel and len(novels_to_train) > 1:
                training_result = self._train_novels_parallel(novels_to_train)
            else:
                # Fallback to sequential training (original implementation)
                logger.info("Using sequential training (parallel disabled or single novel)")
                training_result = self._train_novels_sequential(novels_to_train)

            result["novels_trained"] = training_result["novels_trained"]
            result["training_results"] = training_result["training_results"]
            result["status"] = "completed" if training_result["novels_trained"] > 0 else "failed"

        except Exception as e:
            logger.error(f"Stage 1 failed: {e}")
            result["status"] = "error"
            result["error"] = str(e)

        finally:
            result["duration"] = time.time() - stage_start

        return result

    def stage_2_combined_training(self, model_key: str) -> Dict[str, Any]:
        """Stage 2: Train combined model"""
        logger.info(f"=== Stage 2: Combined Model Training for {model_key} ===")

        stage_start = time.time()
        result = {
            "stage": "combined_training",
            "model_key": model_key,
            "status": "running",
            "start_time": stage_start
        }

        try:
            # Check if combined model already exists
            combined_status = self.check_combined_model_status(model_key)
            result["initial_status"] = combined_status

            if combined_status["model_exists"] and not self.config.force_retrain_combined:
                logger.info("Combined model already exists. Skipping stage 2.")
                result["status"] = "skipped"
                result["reason"] = "model_already_exists"
                return result

            # Run combined model trainer
            cmd = ["python", "combined_model_trainer.py", "--model", model_key]
            if self.config.force_retrain_combined:
                cmd.append("--force")

            logger.info(f"Running: {' '.join(cmd)}")

            process_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
                # No timeout - combined model training can take a long time
            )

            if process_result.returncode == 0:
                result["status"] = "completed"
                result["output"] = process_result.stdout
                logger.info(f"  ✓ Successfully trained combined model: {model_key}")

                # Load training results if available
                results_file = self.models_dir / model_key / "training_results.json"
                if results_file.exists():
                    with open(results_file, 'r', encoding='utf-8') as f:
                        training_data = json.load(f)
                        result["training_data"] = training_data
            else:
                result["status"] = "failed"
                result["error"] = process_result.stderr
                logger.error(f"  ✗ Failed to train combined model: {model_key}")

        # TimeoutExpired exception removed - no timeout for combined training

        except Exception as e:
            logger.error(f"Stage 2 failed: {e}")
            result["status"] = "error"
            result["error"] = str(e)

        finally:
            result["duration"] = time.time() - stage_start

        return result

    def stage_3_relational_mapping(self, model_key: str) -> Dict[str, Any]:
        """Stage 3: Create relational memory mappings"""
        logger.info(f"=== Stage 3: Relational Memory Mapping for {model_key} ===")

        stage_start = time.time()
        result = {
            "stage": "relational_mapping",
            "model_key": model_key,
            "status": "running",
            "start_time": stage_start
        }

        try:
            # Check if mappings already exist
            mappings_status = self.check_relational_mappings_status(model_key)
            result["initial_status"] = mappings_status

            if mappings_status["mappings_exist"] and self.config.skip_if_exists:
                logger.info("Relational mappings already exist. Skipping stage 3.")
                result["status"] = "skipped"
                result["reason"] = "mappings_already_exist"
                return result

            # Run relational memory mapper
            cmd = ["python", "relational_memory_mapper.py", "--model", model_key]

            logger.info(f"Running: {' '.join(cmd)}")

            process_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
                # No timeout - relational mapping can take a long time for large combined models
            )

            if process_result.returncode == 0:
                result["status"] = "completed"
                result["output"] = process_result.stdout
                logger.info(f"  ✓ Successfully created relational mappings: {model_key}")

                # Load mapping results if available
                mappings_file = Path("relational_memories") / f"{model_key}_relational_mappings.json"
                if mappings_file.exists():
                    with open(mappings_file, 'r', encoding='utf-8') as f:
                        mapping_data = json.load(f)
                        result["mapping_data"] = {
                            "novel_count": mapping_data.get("novel_count", 0),
                            "thematic_connections": len(mapping_data.get("thematic_analysis", {}).get("thematic_connections", [])),
                            "character_connections": len(mapping_data.get("character_analysis", {}).get("character_connections", [])),
                            "cross_references": sum(len(refs) for refs in mapping_data.get("cross_references", {}).values())
                        }
            else:
                result["status"] = "failed"
                result["error"] = process_result.stderr
                logger.error(f"  ✗ Failed to create relational mappings: {model_key}")

        # TimeoutExpired exception removed - no timeout for relational mapping

        except Exception as e:
            logger.error(f"Stage 3 failed: {e}")
            result["status"] = "error"
            result["error"] = str(e)

        finally:
            result["duration"] = time.time() - stage_start

        return result

    def stage_4_validation_testing(self, model_key: str) -> Dict[str, Any]:
        """Stage 4: Validate combined model and memory system"""
        logger.info(f"=== Stage 4: Validation Testing for {model_key} ===")

        stage_start = time.time()
        result = {
            "stage": "validation",
            "model_key": model_key,
            "status": "running",
            "start_time": stage_start
        }

        try:
            validation_results = {
                "model_validation": self._validate_combined_model(model_key),
                "memory_validation": self._validate_memory_system(model_key),
                "integration_validation": self._validate_integration(model_key)
            }

            result["validation_results"] = validation_results

            # Determine overall validation status
            all_validations = [v.get("status", "failed") for v in validation_results.values()]
            if all(status == "passed" for status in all_validations):
                result["status"] = "passed"
            elif any(status == "passed" for status in all_validations):
                result["status"] = "partial"
            else:
                result["status"] = "failed"

            logger.info(f"  Validation status: {result['status']}")

        except Exception as e:
            logger.error(f"Stage 4 failed: {e}")
            result["status"] = "error"
            result["error"] = str(e)

        finally:
            result["duration"] = time.time() - stage_start

        return result

    def _validate_combined_model(self, model_key: str) -> Dict[str, Any]:
        """Validate the combined model"""
        model_path = self.models_dir / model_key / "final"

        if not model_path.exists():
            return {"status": "failed", "reason": "Model directory does not exist"}

        # Check for required files
        required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
        missing_files = []

        for filename in required_files:
            if not (model_path / filename).exists():
                missing_files.append(filename)

        if missing_files:
            return {"status": "failed", "reason": f"Missing files: {missing_files}"}

        return {"status": "passed", "model_path": str(model_path)}

    def _validate_memory_system(self, model_key: str) -> Dict[str, Any]:
        """Validate the memory system"""
        memory_path = self.models_dir / model_key / "memory"
        mappings_file = Path("relational_memories") / f"{model_key}_relational_mappings.json"

        validation = {"status": "passed", "checks": []}

        # Check memory directory
        if memory_path.exists():
            validation["checks"].append({"check": "memory_directory", "status": "passed"})
        else:
            validation["checks"].append({"check": "memory_directory", "status": "failed"})

        # Check relational mappings
        if mappings_file.exists():
            validation["checks"].append({"check": "relational_mappings", "status": "passed"})
        else:
            validation["checks"].append({"check": "relational_mappings", "status": "failed"})

        # Overall status
        failed_checks = [c for c in validation["checks"] if c["status"] == "failed"]
        if failed_checks:
            validation["status"] = "partial" if len(failed_checks) < len(validation["checks"]) else "failed"

        return validation

    def _validate_integration(self, model_key: str) -> Dict[str, Any]:
        """Validate integration between components"""
        # Basic integration check - ensure all components reference the same novels
        try:
            novels = self.get_novels_for_model(model_key)
            expected_novel_count = len(novels)

            integration_status = {"status": "passed", "expected_novels": expected_novel_count}

            # Check if training results match expected novel count
            results_file = self.models_dir / model_key / "training_results.json"
            if results_file.exists():
                with open(results_file, 'r', encoding='utf-8') as f:
                    training_data = json.load(f)
                    if "novels_included" in training_data:
                        actual_count = len(training_data["novels_included"])
                        integration_status["actual_novels_in_training"] = actual_count
                        if actual_count != expected_novel_count:
                            integration_status["status"] = "failed"
                            integration_status["reason"] = f"Novel count mismatch: expected {expected_novel_count}, got {actual_count}"

            return integration_status

        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def execute_full_pipeline(self, model_key: str) -> Dict[str, Any]:
        """Execute the complete training pipeline for a model"""
        logger.info(f"\n{'='*60}")
        logger.info(f"STARTING FULL PIPELINE FOR MODEL: {model_key}")
        logger.info(f"{'='*60}")

        self.start_time = time.time()

        pipeline_result = {
            "model_key": model_key,
            "pipeline_start_time": self.start_time,
            "stages": {},
            "overall_status": "running"
        }

        try:
            # Stage 1: Individual novel training
            if self.config.run_individual_training:
                stage1_result = self.stage_1_individual_training(model_key)
                pipeline_result["stages"]["stage_1"] = stage1_result

                if stage1_result["status"] == "error" and self.config.stop_on_error:
                    pipeline_result["overall_status"] = "failed"
                    pipeline_result["failed_at_stage"] = "individual_training"
                    return pipeline_result

            # Stage 2: Combined model training
            if self.config.run_combined_training:
                stage2_result = self.stage_2_combined_training(model_key)
                pipeline_result["stages"]["stage_2"] = stage2_result

                if stage2_result["status"] == "error" and self.config.stop_on_error:
                    pipeline_result["overall_status"] = "failed"
                    pipeline_result["failed_at_stage"] = "combined_training"
                    return pipeline_result

            # Stage 3: Relational mapping
            if self.config.run_relational_mapping:
                stage3_result = self.stage_3_relational_mapping(model_key)
                pipeline_result["stages"]["stage_3"] = stage3_result

                if stage3_result["status"] == "error" and self.config.stop_on_error:
                    pipeline_result["overall_status"] = "failed"
                    pipeline_result["failed_at_stage"] = "relational_mapping"
                    return pipeline_result

            # Stage 4: Validation
            if self.config.run_validation:
                stage4_result = self.stage_4_validation_testing(model_key)
                pipeline_result["stages"]["stage_4"] = stage4_result

            # Determine overall status
            stage_statuses = [stage.get("status", "unknown") for stage in pipeline_result["stages"].values()]

            if all(status in ["completed", "skipped", "passed"] for status in stage_statuses):
                pipeline_result["overall_status"] = "success"
            elif any(status == "error" for status in stage_statuses):
                pipeline_result["overall_status"] = "failed"
            else:
                pipeline_result["overall_status"] = "partial"

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            pipeline_result["overall_status"] = "error"
            pipeline_result["error"] = str(e)

        finally:
            pipeline_result["total_duration"] = time.time() - self.start_time
            logger.info(f"\nPipeline completed for {model_key}: {pipeline_result['overall_status']}")
            logger.info(f"Total duration: {pipeline_result['total_duration']:.1f} seconds")

            # Save pipeline results
            if self.config.save_intermediate_results:
                self._save_pipeline_results(model_key, pipeline_result)

        return pipeline_result

    def _save_pipeline_results(self, model_key: str, pipeline_result: Dict[str, Any]):
        """Save pipeline results to file"""
        results_file = self.results_dir / f"{model_key}_pipeline_results.json"

        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(pipeline_result, f, indent=2, default=str)
            logger.info(f"Pipeline results saved to: {results_file}")
        except Exception as e:
            logger.error(f"Failed to save pipeline results: {e}")

    def execute_all_models_pipeline(self) -> Dict[str, Any]:
        """Execute pipeline for all available models"""
        available_models = self.get_available_models()
        logger.info(f"Executing pipeline for {len(available_models)} models")

        all_results = {
            "pipeline_start_time": time.time(),
            "models": {},
            "summary": {}
        }

        for i, model_key in enumerate(available_models):
            logger.info(f"\n{'='*80}")
            logger.info(f"PROCESSING MODEL {i+1}/{len(available_models)}: {model_key}")
            logger.info(f"{'='*80}")

            try:
                model_result = self.execute_full_pipeline(model_key)
                all_results["models"][model_key] = model_result
            except Exception as e:
                logger.error(f"Failed to process model {model_key}: {e}")
                all_results["models"][model_key] = {
                    "model_key": model_key,
                    "overall_status": "error",
                    "error": str(e)
                }

        # Generate summary
        all_results["total_duration"] = time.time() - all_results["pipeline_start_time"]
        all_results["summary"] = self._generate_summary(all_results["models"])

        # Save overall results
        if self.config.generate_summary_report:
            self._save_summary_report(all_results)

        return all_results

    def _generate_summary(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of all model results"""
        summary = {
            "total_models": len(model_results),
            "successful": 0,
            "failed": 0,
            "partial": 0,
            "errors": 0
        }

        for model_result in model_results.values():
            status = model_result.get("overall_status", "unknown")
            if status == "success":
                summary["successful"] += 1
            elif status == "failed":
                summary["failed"] += 1
            elif status == "partial":
                summary["partial"] += 1
            elif status == "error":
                summary["errors"] += 1

        return summary

    def _save_summary_report(self, all_results: Dict[str, Any]):
        """Save comprehensive summary report"""
        report_file = self.results_dir / "pipeline_summary_report.json"

        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, default=str)
            logger.info(f"Summary report saved to: {report_file}")
        except Exception as e:
            logger.error(f"Failed to save summary report: {e}")


def main():
    """Main entry point for master training pipeline"""
    parser = argparse.ArgumentParser(description="Model Tea - Master Training Pipeline")
    parser.add_argument("--model", type=str, help="Specific model to train")
    parser.add_argument("--all-models", action="store_true", help="Train all models")
    parser.add_argument("--list-models", action="store_true", help="List available models")

    # Pipeline configuration
    parser.add_argument("--skip-individual", action="store_true", help="Skip individual novel training")
    parser.add_argument("--skip-combined", action="store_true", help="Skip combined model training")
    parser.add_argument("--skip-relational", action="store_true", help="Skip relational mapping")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation")

    # Force options
    parser.add_argument("--force-individual", action="store_true", help="Force retrain individual novels")
    parser.add_argument("--force-combined", action="store_true", help="Force retrain combined model")

    # Behavior options
    parser.add_argument("--stop-on-error", action="store_true", help="Stop pipeline on first error")
    parser.add_argument("--no-report", action="store_true", help="Don't generate summary report")

    # Parallel execution options
    parser.add_argument("--parallel", action="store_true", help="Enable parallel novel training")
    parser.add_argument("--max-workers", type=int, default=3, help="Maximum parallel workers (default: 3)")

    args = parser.parse_args()

    try:
        # Create pipeline configuration
        config = PipelineConfig()
        config.run_individual_training = not args.skip_individual
        config.run_combined_training = not args.skip_combined
        config.run_relational_mapping = not args.skip_relational
        config.run_validation = not args.skip_validation
        config.force_retrain_individual = args.force_individual
        config.force_retrain_combined = args.force_combined
        config.stop_on_error = args.stop_on_error
        config.generate_summary_report = not args.no_report
        config.individual_training_parallel = args.parallel
        config.max_parallel_novels = args.max_workers

        # Initialize pipeline
        pipeline = MasterTrainingPipeline(config)

        if args.list_models:
            models = pipeline.get_available_models()
            print(f"\nAvailable Models ({len(models)}):")
            for model in models:
                print(f"  - {model}")
            return

        if args.all_models:
            print("Executing full pipeline for all models...")
            results = pipeline.execute_all_models_pipeline()

            # Display summary
            summary = results["summary"]
            print(f"\n=== PIPELINE SUMMARY ===")
            print(f"Total models: {summary['total_models']}")
            print(f"Successful: {summary['successful']}")
            print(f"Failed: {summary['failed']}")
            print(f"Partial: {summary['partial']}")
            print(f"Errors: {summary['errors']}")
            print(f"Total duration: {results['total_duration']:.1f} seconds")

        elif args.model:
            print(f"Executing full pipeline for: {args.model}")
            result = pipeline.execute_full_pipeline(args.model)
            print(f"\nPipeline status: {result['overall_status']}")
            print(f"Duration: {result.get('total_duration', 0):.1f} seconds")

        else:
            # No specific arguments - show help
            parser.print_help()

            # Show available models
            models = pipeline.get_available_models()
            print(f"\nAvailable models: {', '.join(models)}")
            print(f"\nExample usage:")
            print(f"  python master_training_pipeline.py --model MODEL_NAME")
            print(f"  python master_training_pipeline.py --model MODEL_NAME --parallel")
            print(f"  python master_training_pipeline.py --all-models --parallel --max-workers 4")

    except Exception as e:
        logger.error(f"Master pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
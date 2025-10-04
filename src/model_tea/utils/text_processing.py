import numpy as np
from typing import List, Dict, Any


class TextProcessingUtils:
    @staticmethod
    def create_progressive_chunks(text: str, iteration: int, base_size: int = 200) -> List[str]:
        progression_factor = 1 + (iteration * 0.2)
        current_size = int(base_size * progression_factor)

        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]

        chunks = []
        current_chunk = ""
        current_words = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            if current_words + sentence_words > current_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_words = sentence_words
            else:
                current_chunk += " " + sentence
                current_words += sentence_words

        if current_chunk and current_words > 10:
            chunks.append(current_chunk.strip())

        return chunks

    @staticmethod
    def calculate_text_stats(text: str) -> Dict[str, Any]:
        words = text.split()
        sentences = [s for s in text.split('.') if s.strip()]

        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_words_per_sentence": len(words) / len(sentences) if sentences else 0,
            "character_count": len(text),
            "estimated_reading_time": len(words) / 250
        }


class QualityMetrics:
    @staticmethod
    def assess_text_quality(text: str) -> Dict[str, float]:
        words = text.split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        unique_words = set(word.lower() for word in words)
        repetition_ratio = len(unique_words) / len(words) if words else 0

        sentence_lengths = [len(s.split()) for s in sentences]
        avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
        sentence_variance = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0

        quality_score = (
            repetition_ratio * 0.4 +
            min(1.0, avg_sentence_length / 15) * 0.3 +
            min(1.0, sentence_variance / 25) * 0.3
        )

        return {
            "quality_score": quality_score,
            "repetition_ratio": repetition_ratio,
            "avg_sentence_length": avg_sentence_length,
            "sentence_variance": sentence_variance,
            "vocabulary_diversity": len(unique_words),
            "total_words": len(words)
        }

    @staticmethod
    def calculate_perplexity_from_loss(loss: float) -> float:
        return np.exp(loss)

    @staticmethod
    def assess_generation_improvement(baseline: str, enhanced: str) -> Dict[str, float]:
        baseline_metrics = QualityMetrics.assess_text_quality(baseline)
        enhanced_metrics = QualityMetrics.assess_text_quality(enhanced)

        return {
            "quality_improvement": enhanced_metrics["quality_score"] - baseline_metrics["quality_score"],
            "vocabulary_improvement": enhanced_metrics["vocabulary_diversity"] - baseline_metrics["vocabulary_diversity"],
            "length_change": enhanced_metrics["total_words"] - baseline_metrics["total_words"],
            "baseline_quality": baseline_metrics["quality_score"],
            "enhanced_quality": enhanced_metrics["quality_score"]
        }

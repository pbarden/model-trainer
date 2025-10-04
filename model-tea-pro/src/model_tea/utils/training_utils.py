class TrainingUtils:
    @staticmethod
    def calculate_learning_rate(iteration: int, total_iterations: int,
                              start_lr: float = 5e-5, end_lr: float = 1e-5) -> float:
        if total_iterations <= 1:
            return start_lr

        progress = iteration / (total_iterations - 1)
        return start_lr * (1 - progress) + end_lr * progress

    @staticmethod
    def calculate_chunk_size(iteration: int, base_size: int = 200) -> int:
        return int(base_size + (iteration * base_size * 0.2))

    @staticmethod
    def format_training_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from model_tea.services import MetadataService

console = Console()


@click.command()
@click.argument("model_key")
@click.option("--max-iterations", type=int, help="Maximum training iterations")
def train_model(model_key, max_iterations):
    """Train a model from models.json (primary method)"""
    from model_tea.trainers.model import ModelConfig
    from model_tea.services import get_training_service

    config = ModelConfig()

    if max_iterations:
        config.max_iterations = max_iterations

    TrainingService = get_training_service()
    service = TrainingService()

    console.print(f"\n[bold cyan]Training model:[/bold cyan] {model_key}\n")

    try:
        results = service.train_model(model_key, config)
        console.print("\n[bold green]Training completed successfully![/bold green]\n")
    except Exception as e:
        console.print(f"\n[bold red]Training failed:[/bold red] {e}\n")
        raise click.Abort()


@click.command()
@click.argument("novel_name")
@click.option("--max-iterations", type=int, help="Maximum training iterations")
@click.option("--batch-size", type=int, help="Training batch size")
@click.option("--learning-rate", type=float, help="Starting learning rate")
def train_direct(novel_name, max_iterations, batch_size, learning_rate):
    """Train a novel directly from novels/ directory (dev/testing)"""
    from model_tea.trainers.iterative import IterativeConfig
    from model_tea.services import get_training_service

    config = IterativeConfig()

    if max_iterations:
        config.max_iterations = max_iterations
    if batch_size:
        config.batch_size = batch_size
    if learning_rate:
        config.learning_rate_start = learning_rate

    TrainingService = get_training_service()
    service = TrainingService()

    console.print(f"\n[bold cyan]Training novel directly:[/bold cyan] {novel_name}")
    console.print(f"[dim]Max iterations: {config.max_iterations}[/dim]")
    console.print(f"[dim]Batch size: {config.batch_size}[/dim]\n")

    try:
        results = service.train_direct(novel_name, config)
        console.print("\n[bold green]Training completed successfully![/bold green]\n")

        table = Table(title="Training Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Total Time", f"{results.get('training_time', 0):.1f}s")
        table.add_row("Iterations", str(len(results.get('iterations', []))))
        table.add_row("Final Quality", f"{results.get('final_quality', 0):.3f}")

        console.print(table)

    except Exception as e:
        console.print(f"\n[bold red]Training failed:[/bold red] {e}\n")
        raise click.Abort()


@click.command()
def list_novels():
    service = MetadataService()
    novels = service.list_novels()

    if not novels:
        console.print("[yellow]No novels found in novels.json[/yellow]")
        return

    table = Table(title="Available Novels")
    table.add_column("Novel", style="cyan")
    table.add_column("Words", style="magenta", justify="right")

    for novel in novels:
        table.add_row(novel["name"], f"{novel['word_count']:,}")

    console.print(table)
    console.print(f"\n[dim]Total: {len(novels)} novels[/dim]\n")

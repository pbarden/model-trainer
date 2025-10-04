import click
from rich.console import Console
from rich.table import Table
from pathlib import Path
import shutil

from model_tea.services import ChatService

console = Console()


@click.command()
@click.option("--all", "-a", is_flag=True, help="Show all models including iterations")
def list_models(all):
    service = ChatService()
    models = service.list_available_models(show_all=all)

    if not models:
        console.print("[yellow]No models found in iterative_models/ directory[/yellow]")
        return

    table = Table(title="Available Models")
    table.add_column("Model", style="cyan")

    for model in models:
        table.add_row(model)

    console.print(table)
    console.print(f"\n[dim]Total: {len(models)} models[/dim]\n")


@click.command()
@click.argument("model_name")
def model_info(model_name):
    service = ChatService()
    info = service.get_model_info(model_name)

    if not info["exists"]:
        console.print(f"[bold red]Model not found:[/bold red] {model_name}")
        return

    table = Table(title=f"Model Info: {model_name}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Path", info["path"])
    table.add_row("Size (MB)", str(info.get("size_mb", "N/A")))
    table.add_row("Type", info.get("model_type", "N/A"))
    table.add_row("Vocab Size", str(info.get("vocab_size", "N/A")))

    console.print(table)


@click.command()
@click.argument("model_name")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
def delete_model(model_name, force):
    models_dir = Path("iterative_models")
    model_path = models_dir / model_name

    if not model_path.exists():
        console.print(f"[bold red]Model not found:[/bold red] {model_name}")
        return

    if not force:
        confirm = click.confirm(f"Delete model '{model_name}'?")
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            return

    try:
        shutil.rmtree(model_path)
        console.print(f"[bold green]Deleted:[/bold green] {model_name}")
    except Exception as e:
        console.print(f"[bold red]Failed to delete:[/bold red] {e}")

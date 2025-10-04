import click
from rich.console import Console
from rich.table import Table
from pathlib import Path
import shutil

from model_tea.services import MetadataService

console = Console()


@click.command()
@click.option("--trained", "-t", is_flag=True, help="Show only trained models")
def list_models(trained):
    service = MetadataService()

    if trained:
        models = service.list_trained_models()
        if not models:
            console.print("[yellow]No trained models found[/yellow]")
            return

        table = Table(title="Trained Models")
        table.add_column("Model", style="cyan")
        table.add_column("Size (MB)", style="magenta", justify="right")

        for model in models:
            table.add_row(model["key"], str(model["size_mb"]))

        console.print(table)
        console.print(f"\n[dim]Total: {len(models)} trained models[/dim]\n")

    else:
        models = service.list_models()
        if not models:
            console.print("[yellow]No models found in models.json[/yellow]")
            return

        table = Table(title="Available Models")
        table.add_column("Model", style="cyan")
        table.add_column("Novels", style="blue", justify="right")
        table.add_column("Words", style="magenta", justify="right")
        table.add_column("Trained", style="green")

        trained_keys = {m["key"] for m in service.list_trained_models()}

        for model in models:
            is_trained = "Yes" if model["key"] in trained_keys else ""
            table.add_row(
                model["key"],
                str(model["novel_count"]),
                f"{model['total_words']:,}",
                is_trained
            )

        console.print(table)
        console.print(f"\n[dim]Total: {len(models)} models ({len(trained_keys)} trained)[/dim]\n")


@click.command()
@click.argument("model_name")
def model_info(model_name):
    service = MetadataService()
    info = service.get_model_info(model_name)

    if not info:
        console.print(f"[bold red]Model not found in models.json:[/bold red] {model_name}")
        return

    table = Table(title=f"Model Info: {model_name}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Novels", str(info["novel_count"]))
    table.add_row("Total Words", f"{info['total_words']:,}")
    table.add_row("Trained", "Yes" if info["is_trained"] else "No")

    if info["is_trained"]:
        table.add_row("Trained Path", info["trained_path"])
        table.add_row("Size (MB)", str(info["size_mb"]))

    console.print(table)

    if info["novels"]:
        console.print("\n[bold cyan]Novels:[/bold cyan]")
        for novel in info["novels"]:
            console.print(f"  • {novel['name']} ({novel['word_count']:,} words)")


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

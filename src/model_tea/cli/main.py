import click
import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel

from model_tea.services import MetadataService

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)


def show_overview():
    """Show system overview"""
    service = MetadataService()
    status = service.get_system_status()

    console.print("\n[bold cyan]Model Tea[/bold cyan] - Language Model Training System\n")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Novels", str(status["novels_count"]))
    table.add_row("Models", str(status["models_count"]))
    table.add_row("Trained", str(status["trained_models_count"]))

    console.print(table)

    console.print("\n[dim]Commands:[/dim]")
    console.print("  [cyan]model-tea status[/cyan]       - Detailed system status")
    console.print("  [cyan]model-tea list[/cyan]         - List resources")
    console.print("  [cyan]model-tea train <key>[/cyan]  - Train a model")
    console.print("  [cyan]model-tea chat <model>[/cyan] - Chat with a model")
    console.print("  [cyan]model-tea --help[/cyan]       - Show all commands\n")


@click.group(invoke_without_command=True)
@click.pass_context
@click.version_option(version="2.0.0")
def cli(ctx):
    if ctx.invoked_subcommand is None:
        show_overview()


@cli.command()
def status():
    """Show detailed system status"""
    service = MetadataService()
    status_info = service.get_system_status()

    table = Table(title="System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Count", style="magenta", justify="right")

    table.add_row(
        "novels.json",
        "OK" if status_info["novels_file_exists"] else "MISSING",
        str(status_info["novels_count"])
    )
    table.add_row(
        "models.json",
        "OK" if status_info["models_file_exists"] else "MISSING",
        str(status_info["models_count"])
    )
    table.add_row(
        "novels/ directory",
        "OK" if status_info["novels_dir_exists"] else "MISSING",
        "-"
    )
    table.add_row(
        "iterative_models/ directory",
        "OK" if status_info["models_dir_exists"] else "MISSING",
        str(status_info["trained_models_count"]) + " trained"
    )

    console.print(table)


@cli.command()
@click.option("--novels", "-n", is_flag=True, help="List novels")
@click.option("--models", "-m", is_flag=True, help="List models")
@click.option("--trained", "-t", is_flag=True, help="List trained models")
def list(novels, models, trained):
    """List resources (novels, models, trained)"""
    service = MetadataService()

    if not (novels or models or trained):
        novels = models = trained = True

    if novels:
        novel_list = service.list_novels()
        table = Table(title="Novels")
        table.add_column("Novel", style="cyan")
        table.add_column("Words", style="magenta", justify="right")

        for novel in novel_list[:20]:
            table.add_row(novel["name"], f"{novel['word_count']:,}")

        console.print(table)
        if len(novel_list) > 20:
            console.print(f"[dim]Showing 20 of {len(novel_list)} novels[/dim]\n")

    if models:
        model_list = service.list_models()
        trained_keys = {m["key"] for m in service.list_trained_models()}

        table = Table(title="Models")
        table.add_column("Model", style="cyan")
        table.add_column("Novels", style="blue", justify="right")
        table.add_column("Words", style="magenta", justify="right")
        table.add_column("Trained", style="green")

        for model in model_list[:20]:
            is_trained = "Yes" if model["key"] in trained_keys else ""
            table.add_row(
                model["key"],
                str(model["novel_count"]),
                f"{model['total_words']:,}",
                is_trained
            )

        console.print(table)
        if len(model_list) > 20:
            console.print(f"[dim]Showing 20 of {len(model_list)} models[/dim]\n")

    if trained:
        trained_list = service.list_trained_models()
        table = Table(title="Trained Models")
        table.add_column("Model", style="cyan")
        table.add_column("Size (MB)", style="magenta", justify="right")

        for model in trained_list[:20]:
            table.add_row(model["key"], str(model["size_mb"]))

        console.print(table)
        if len(trained_list) > 20:
            console.print(f"[dim]Showing 20 of {len(trained_list)} trained models[/dim]\n")


@cli.group()
def train():
    pass


@cli.group()
def chat():
    pass


@cli.group()
def models():
    pass


from .train_cmd import train_model, train_direct, list_novels
from .chat_cmd import chat_interactive, chat_generate
from .models_cmd import list_models, model_info, delete_model

train.add_command(train_model)
train.add_command(train_direct, name="direct")
train.add_command(list_novels, name="list")

chat.add_command(chat_interactive, name="interactive")
chat.add_command(chat_generate, name="generate")

models.add_command(list_models, name="list")
models.add_command(model_info, name="info")
models.add_command(delete_model, name="delete")


if __name__ == "__main__":
    cli()

import click
import logging
from rich.console import Console
from rich.logging import RichHandler

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)


@click.group()
@click.version_option(version="2.0.0")
def cli():
    pass


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

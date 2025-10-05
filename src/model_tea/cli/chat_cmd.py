import click
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

console = Console()


@click.command()
@click.argument("model_name")
@click.option("--max-length", type=int, help="Maximum generation length")
@click.option("--temperature", type=float, help="Temperature for generation")
def chat_interactive(model_name, max_length, temperature):
    from model_tea.services import get_chat_service
    from model_tea.config.settings import settings

    max_length = max_length or settings.default_max_length
    temperature = temperature or settings.default_temperature

    ChatService = get_chat_service()
    service = ChatService()

    console.print(f"\n[bold cyan]Loading model:[/bold cyan] {model_name}\n")

    if not service.load_model(model_name):
        console.print("[bold red]Failed to load model[/bold red]")
        raise click.Abort()

    console.print("[bold green]Model loaded successfully![/bold green]")
    console.print("[dim]Type your prompt and press Enter. Type 'quit' to exit.[/dim]\n")

    while True:
        try:
            prompt = Prompt.ask("\n[bold cyan]You[/bold cyan]")

            if prompt.lower() in ["quit", "exit", "q"]:
                console.print("\n[yellow]Goodbye![/yellow]\n")
                break

            if not prompt.strip():
                continue

            response = service.generate(
                prompt,
                max_length=max_length,
                temperature=temperature
            )

            console.print(Panel(
                response,
                title="[bold magenta]Model[/bold magenta]",
                border_style="magenta"
            ))

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Interrupted. Goodbye![/yellow]\n")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}\n")


@click.command()
@click.argument("model_name")
@click.option("--prompt", "-p", required=True, help="Text prompt for generation")
@click.option("--max-length", type=int, help="Maximum generation length")
@click.option("--temperature", type=float, help="Temperature for generation")
def chat_generate(model_name, prompt, max_length, temperature):
    from model_tea.services import get_chat_service
    from model_tea.config.settings import settings

    max_length = max_length or settings.default_max_length
    temperature = temperature or settings.default_temperature

    ChatService = get_chat_service()
    service = ChatService()

    if not service.load_model(model_name):
        console.print("[bold red]Failed to load model[/bold red]")
        raise click.Abort()

    try:
        response = service.generate(
            prompt,
            max_length=max_length,
            temperature=temperature
        )

        console.print("\n" + response + "\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        raise click.Abort()

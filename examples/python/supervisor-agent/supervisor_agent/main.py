"""CLI interface for the Supervisor Agent system."""

import sys
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from supervisor_agent.supervisor.supervisor import SupervisorAgent
from supervisor_agent.utils.config import get_config, validate_config


console = Console()


def display_welcome():
    """Display welcome message and system information."""
    welcome_text = Text()
    welcome_text.append("🤖 Supervisor Agent System\n", style="bold blue")
    welcome_text.append("Multi-Agent AI Assistant with Specialized Capabilities\n\n", style="italic")
    welcome_text.append("Available Agents:\n", style="bold")
    welcome_text.append("• Code Agent: Code generation, debugging, testing\n", style="green")
    welcome_text.append("• Data Agent: Data analysis, statistics, visualizations\n", style="cyan") 
    welcome_text.append("• File Agent: File operations, directory management\n", style="yellow")
    
    console.print(Panel(welcome_text, title="Welcome", border_style="blue"))


def display_agent_status(supervisor: SupervisorAgent):
    """Display the status of all agents."""
    health = supervisor.health_check()
    capabilities = supervisor.get_agent_capabilities()
    
    # Create status table
    table = Table(title="Agent Status", show_header=True, header_style="bold magenta")
    table.add_column("Agent", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    table.add_column("Description", style="white")
    
    status_map = {
        "supervisor": health["supervisor"],
        "code_agent": health["code_agent"],
        "data_agent": health["data_agent"],
        "file_agent": health["file_agent"]
    }
    
    for agent, status in status_map.items():
        status_text = "✅ Ready" if status else "❌ Error"
        description = capabilities.get(agent, "Unknown")
        table.add_row(agent.replace("_", " ").title(), status_text, description)
    
    console.print(table)
    console.print(f"\nTotal Tools Available: {health['total_tools']}")
    console.print(f"Configuration Valid: {'✅' if health['config_valid'] else '❌'}")


def run_interactive_mode(supervisor: SupervisorAgent):
    """Run the interactive chat mode."""
    console.print("\n[bold green]Interactive Mode Started[/bold green]")
    console.print("Type 'exit', 'quit', or 'bye' to end the session")
    console.print("Type 'status' to see agent status")
    console.print("Type 'help' for available commands\n")
    
    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
            
            if user_input.lower() in ["exit", "quit", "bye"]:
                console.print("[yellow]Goodbye! 👋[/yellow]")
                break
            
            if user_input.lower() == "status":
                display_agent_status(supervisor)
                continue
            
            if user_input.lower() == "help":
                show_help()
                continue
            
            if not user_input.strip():
                continue
            
            # Process the request
            console.print("[bold yellow]Processing...[/bold yellow]")
            
            # Show which agents can handle the request
            capabilities = supervisor.can_handle_request(user_input)
            capable_agents = [agent for agent, can_handle in capabilities.items() if can_handle]
            
            if capable_agents:
                console.print(f"[dim]Routing to: {', '.join(capable_agents)}[/dim]")
            
            # Get response from supervisor
            response = supervisor.process_request(user_input)
            
            # Display response
            console.print(Panel(response, title="Response", border_style="green"))
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Session interrupted. Goodbye! 👋[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")


def show_help():
    """Display help information."""
    help_text = Text()
    help_text.append("Available Commands:\n\n", style="bold")
    help_text.append("General Commands:\n", style="bold cyan")
    help_text.append("• help - Show this help message\n")
    help_text.append("• status - Show agent status and health\n")
    help_text.append("• exit/quit/bye - Exit the program\n\n")
    
    help_text.append("Example Requests:\n", style="bold cyan")
    help_text.append("• 'Generate a Python function to sort a list'\n", style="green") 
    help_text.append("• 'Analyze this CSV data and create a chart'\n", style="cyan")
    help_text.append("• 'Read the contents of config.json'\n", style="yellow")
    help_text.append("• 'Debug this code that has a syntax error'\n", style="green")
    help_text.append("• 'Calculate the mean and median of these numbers'\n", style="cyan")
    help_text.append("• 'List all Python files in the current directory'\n", style="yellow")
    
    console.print(Panel(help_text, title="Help", border_style="blue"))


@click.command()
@click.argument("request", required=False)
@click.option("--interactive", "-i", is_flag=True, help="Start interactive chat mode")
@click.option("--status", "-s", is_flag=True, help="Show agent status and exit")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--config-check", is_flag=True, help="Check configuration and exit")
def main(request: Optional[str], interactive: bool, status: bool, verbose: bool, config_check: bool):
    """Supervisor Agent - Multi-Agent AI Assistant.
    
    Examples:
        supervisor-agent "Generate Python code to read a CSV file"
        supervisor-agent --interactive
        supervisor-agent --status
    """
    
    # Check configuration first
    if not validate_config():
        console.print("[red]❌ Configuration Error:[/red]")
        console.print("OpenAI API key not found. Please set OPENAI_API_KEY environment variable")
        console.print("or create a .env file with OPENAI_API_KEY=your_key_here")
        sys.exit(1)
    
    if config_check:
        console.print("[green]✅ Configuration is valid[/green]")
        config = get_config()
        console.print(f"Model: {config.openai_model}")
        console.print(f"Temperature: {config.temperature}")
        sys.exit(0)
    
    # Initialize supervisor
    try:
        console.print("[yellow]Initializing Supervisor Agent...[/yellow]")
        supervisor = SupervisorAgent()
        console.print("[green]✅ Supervisor Agent initialized successfully[/green]")
    except Exception as e:
        console.print(f"[red]❌ Failed to initialize supervisor: {str(e)}[/red]")
        sys.exit(1)
    
    # Handle status flag
    if status:
        display_agent_status(supervisor)
        sys.exit(0)
    
    # Display welcome message
    if not request:
        display_welcome()
    
    # Handle interactive mode
    if interactive or not request:
        run_interactive_mode(supervisor)
        return
    
    # Handle direct request
    try:
        console.print(f"[bold cyan]Request:[/bold cyan] {request}")
        
        # Show which agents can handle the request
        if verbose:
            capabilities = supervisor.can_handle_request(request)
            capable_agents = [agent for agent, can_handle in capabilities.items() if can_handle]
            if capable_agents:
                console.print(f"[dim]Routing to: {', '.join(capable_agents)}[/dim]")
        
        console.print("[yellow]Processing...[/yellow]")
        
        # Process the request
        response = supervisor.process_request(request)
        
        # Display response
        console.print(Panel(response, title="Response", border_style="green"))
        
    except Exception as e:
        console.print(f"[red]Error processing request: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
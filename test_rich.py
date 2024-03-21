from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.columns import Columns

console = Console()

table1 = Table()

table1.add_column("a")
table1.add_column("b")

table1.add_row("a", "b")
table1.add_row("a", "b")

table2 = Table()

table2.add_column("c")
table2.add_column("d")

table2.add_row("c",  "d")
table2.add_row("c",  "d")

panel = Panel.fit(
    Columns([table1, table2]),
    title="My Panel",
    border_style="red",
    title_align="left",
    padding=(1, 2),
)

console.print(panel)
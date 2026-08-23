import time

from console import console
from rich.progress import track
from rich.prompt import Prompt
from rich.tree import Tree

console.print([1, 2, 3])
console.print("[blue underline]Looks like a link")
# console.print(locals())
console.print("FOO", style="white on blue")

for i in track(range(20), description="Processing..."):
    time.sleep(1)  # Simulate work being done

# console.print(list(SPINNERS.keys()))

pizza_name = Prompt.ask("Enter your pizza name")

with console.status(f"Making {pizza_name}", spinner="shark") as status:
    time.sleep(1)
    status.update("Adding Toppings")
    time.sleep(1)
    status.update("Adding Cheese")
    time.sleep(1)

console.print("Done")

tree = Tree("Rich Tree")
tree.add("Apple")
tree.add("Watermelon")
console.print(tree)

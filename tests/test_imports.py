import importlib

MODULES = [
    "src.core.registry",
    "src.core.types",
    "src.data.preprocessing",
    "src.data.dataset",
    "src.models.encoder",
    "src.models.combiner",
    "src.models.projection",
    "src.models.tower",
    "src.models.two_tower",
    "src.models.retrieval",
]

for module in MODULES:
    print(f"Importing {module} ...", end=" ")
    importlib.import_module(module)
    print("OK")

print("\nAll imports passed.")
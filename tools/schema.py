"""What the tools share: the materials a design may use, and the JSON schema the engineer model sees."""

# The materials the DFT and MD tools compute. A layer is a crystalline ceramic and an interlayer one of the weak phases.
CERAMIC = {"type": "string", "enum": ["alumina", "silicon carbide"]}
INTERLAYER = {"type": "string", "enum": ["silica glass", "graphite", "epoxy", "PMMA"]}


def spec(name, description, properties):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": list(properties)}}}


def number(description):
    return {"type": "number", "description": description}

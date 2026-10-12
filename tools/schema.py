"""What the tools share: the materials a design may use, and the JSON schema the engineer model sees."""

# Approximate surface energies in J/m^2, used only by the stand-ins. The keys are the allowed materials.
SURFACE_ENERGY = {"alumina": 1.5, "silicon carbide": 1.8, "silica glass": 0.3,
                  "graphite": 0.17, "epoxy": 0.045, "PMMA": 0.041}
MATERIAL = {"type": "string", "enum": list(SURFACE_ENERGY)}


def spec(name, description, properties):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": list(properties)}}}


def number(description):
    return {"type": "number", "description": description}

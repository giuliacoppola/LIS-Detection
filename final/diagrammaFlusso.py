import matplotlib.pyplot as plt
import networkx as nx

# Crea grafo
G = nx.DiGraph()

# Nodi
G.add_nodes_from([
    "Acquisizione immagine (PC)",
    "Estrazione landmark (MediaPipe)",
    "Quantizzazione input (int8)",
    "Trasmissione UART → STM32",
    "Parsing input (STM32)",
    "Inferenza STM32Cube.AI",
    "Trasmissione risultati → PC",
    "Decodifica output (PC)",
    "Visualizzazione gesto"
])

# Collegamenti
edges = [
    ("Acquisizione immagine (PC)", "Estrazione landmark (MediaPipe)"),
    ("Estrazione landmark (MediaPipe)", "Quantizzazione input (int8)"),
    ("Quantizzazione input (int8)", "Trasmissione UART → STM32"),
    ("Trasmissione UART → STM32", "Parsing input (STM32)"),
    ("Parsing input (STM32)", "Inferenza STM32Cube.AI"),
    ("Inferenza STM32Cube.AI", "Trasmissione risultati → PC"),
    ("Trasmissione risultati → PC", "Decodifica output (PC)"),
    ("Decodifica output (PC)", "Visualizzazione gesto")
]

G.add_edges_from(edges)

# Disegna
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(14, 8))
nx.draw(G, pos, with_labels=True, node_color="#cce5ff", node_size=4000, font_size=9, font_weight="bold", arrows=True)
plt.title("Diagramma di flusso dell’implementazione software (PC ↔ STM32N6)", fontsize=14)
plt.show()

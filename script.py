import base64
import os
from collections import defaultdict

import clip
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pyvis.network import Network
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from tqdm import tqdm

# === Parameters ===
IMAGE_DIR = "./images"
GROUND_TRUTH_PATH = "./Stempelliste_bueschel_Neuses_einfach.xlsx"

TOP_K_EDGES = 5
TOP_K_PREDICTIONS = 3
HIDDEN_DIM = 128
TEMPERATURE = 0.1
EPOCHS = 250
NUM_NEGATIVES = 30
GNN_LAYERS = 3
GT_WEIGHT = 2
SIM_WEIGHT = 1

# === Load CLIP Model ===
clip_model, preprocess = clip.load("ViT-B/32", device="cpu")


# === Embedding Extraction ===
def extract_clip_embeddings(image_dir):
    image_paths = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    embeddings = []
    filenames = []

    for fname in tqdm(image_paths, desc="Extracting CLIP embeddings"):
        img = Image.open(os.path.join(image_dir, fname))
        if img.mode != 'RGB':
            img = img.convert("RGB")
        image_input = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            emb = clip_model.encode_image(image_input).squeeze(0)
        emb = F.normalize(emb, dim=0)
        embeddings.append(emb)
        filenames.append(fname)

    return torch.stack(embeddings), filenames


# === Ground Truth ===
def split_sequences_for_training(path, filenames, train_fraction=0.66):
    name_to_idx = {name: i for i, name in enumerate(filenames)}
    training_pairs = []
    prediction_tasks = []
    df = pd.read_excel(path)

    def process_chain(group, suffix):
        ids = group[df.columns[0]].astype(str) + suffix + ".jpg"
        idxs = [name_to_idx[id_] for id_ in ids if id_ in name_to_idx]
        if len(idxs) < 2:
            return

        split_point = int(len(idxs) * train_fraction)
        train_idxs = idxs[:split_point]
        predict_idxs = idxs[split_point:]

        for i in range(len(train_idxs) - 1):
            training_pairs.append((train_idxs[i], train_idxs[i + 1]))

        if predict_idxs:
            prediction_tasks.append((train_idxs[-1], predict_idxs))

    grouped_vorder = df.groupby(df.columns[1])
    grouped_rueck = df.groupby(df.columns[2])

    for _, group in grouped_vorder:
        process_chain(group, "_a")
    for _, group in grouped_rueck:
        process_chain(group, "_r")

    return training_pairs, prediction_tasks


# === Graph Construction ===
def build_weighted_hybrid_graph(embeddings, ground_truth_pairs, k, gt_weight, sim_weight):
    sim = cosine_similarity(embeddings.numpy())
    edge_index = []
    edge_weight = []
    N = len(embeddings)

    for a, b in ground_truth_pairs:
        edge_index.append([a, b])  # Directed edge
        edge_weight.append(gt_weight)

    for i in range(N):
        top_k = np.argsort(sim[i])[-(k + 1):]
        for j in top_k:
            if i != j:
                edge_index.append([i, j])
                edge_weight.append(sim[i, j] * sim_weight)

    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    return edge_index, edge_weight


# === GNN Model ===
class GNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, heads=4):
        super().__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=0.1))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.1))

        # Output layer
        self.layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=0.1))

    def forward(self, x, edge_index):
        for conv in self.layers[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
        x = self.layers[-1](x, edge_index)
        return x


# === Training ===

def sample_hard_negatives(anchor_idx, embeddings, num_negatives, exclude_indices):
    sim = cosine_similarity(
        embeddings[anchor_idx].unsqueeze(0).numpy(), embeddings.numpy()
    )[0]

    for idx in exclude_indices:
        sim[idx] = -1

    hard_negatives = np.argsort(sim)[-num_negatives:]
    return torch.tensor(hard_negatives, dtype=torch.long)


def train_sequence_model_with_hard_negatives(data, chains, hidden_dim, epochs, temperature, num_negatives):
    encoder = GNNEncoder(data.num_node_features, hidden_dim, GNN_LAYERS)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

    for epoch in range(epochs):
        encoder.train()
        h = encoder(data.x, data.edge_index)
        h = F.normalize(h, dim=1)
        total_loss = 0

        for a, b in chains:
            anchor = h[a]
            positive = h[b]

            negatives_idx = sample_hard_negatives(
                anchor_idx=a,
                embeddings=data.x,
                num_negatives=num_negatives,
                exclude_indices={a, b}
            )
            negatives = h[negatives_idx]

            pos_sim = torch.matmul(anchor, positive) / temperature
            neg_sims = torch.matmul(negatives, anchor) / temperature
            logits = torch.cat([pos_sim.unsqueeze(0), neg_sims], dim=0)
            labels = torch.zeros(1, dtype=torch.long)

            loss = F.cross_entropy(logits.unsqueeze(0), labels)
            total_loss += loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss.item():.4f}")

    return encoder


# === Prediction & Evaluation ===

def stepwise_sequence_prediction(data, encoder, filenames, prediction_tasks, top_k):
    encoder.eval()
    with torch.no_grad():
        h = encoder(data.x, data.edge_index)
        h = F.normalize(h, dim=1)
        results = []

        for start_node, future_sequence in prediction_tasks:
            current_node = start_node
            sequence_hits = []

            for true_next in future_sequence:
                sims = torch.matmul(h, h[current_node])
                sims[current_node] = -float('inf')
                top_indices = torch.topk(sims, top_k).indices.tolist()
                hit = true_next in top_indices
                sequence_hits.append(hit)
                print(f"{filenames[current_node]} ➔ {filenames[true_next]} | {'Correct' if hit else 'Incorrect'}")
                current_node = true_next

            results.append(sequence_hits)

        return results


def predict_next_in_sequence(data, encoder, filenames, ground_truth_pairs, top_k):
    encoder.eval()
    with torch.no_grad():
        h = encoder(data.x, data.edge_index)
        h = F.normalize(h, dim=1)

        # Build sequences from ground truth
        successors = defaultdict(list)
        predecessors = defaultdict(list)
        for src, tgt in ground_truth_pairs:
            successors[src].append(tgt)
            predecessors[tgt].append(src)

        all_gt_nodes = set([n for pair in ground_truth_pairs for n in pair])
        start_nodes = set(node for node in all_gt_nodes if node not in predecessors)
        end_nodes = set(node for node in all_gt_nodes if node not in successors)
        middle_nodes = all_gt_nodes - start_nodes - end_nodes

        ground_truth_set = set(ground_truth_pairs)
        predictions = {}

        for idx in range(h.size(0)):
            anchor = h[idx]
            sims = torch.matmul(h, anchor)
            sims[idx] = -float('inf')  # Avoid self-loop

            is_in_gt = idx in all_gt_nodes

            # Completely block start and middle nodes from predicting
            if is_in_gt and (idx in start_nodes or idx in middle_nodes):
                predictions[filenames[idx]] = []
                continue  # Skip to next node

            for target_idx in range(h.size(0)):
                # Block ground truth edges
                if (idx, target_idx) in ground_truth_set:
                    sims[target_idx] = -float('inf')

                # Block predictions INTO middle or end nodes
                if target_idx in middle_nodes or target_idx in end_nodes:
                    sims[target_idx] = -float('inf')

                # Block external coins predicting INTO ground truth (except start nodes)
                if (not is_in_gt) and (target_idx in all_gt_nodes) and (target_idx not in start_nodes):
                    sims[target_idx] = -float('inf')

            top_indices = torch.topk(sims, top_k).indices.tolist()
            predictions[filenames[idx]] = [filenames[i] for i in top_indices]

        return predictions


def evaluate_stepwise_predictions(results):
    total_steps = sum(len(seq) for seq in results)
    total_hits = sum(hit for seq in results for hit in seq)
    accuracy = total_hits / total_steps if total_steps > 0 else 0
    print(f"\nStepwise Sequence Prediction Accuracy: {accuracy:.4f}")
    return accuracy


# === Visualization ===

def image_to_data_url(filepath, size=150):
    with open(filepath, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
        return f'<img src="data:image/jpeg;base64,{encoded}" width="{size}">'


def visualize_graph_with_image_hover(gt_pairs, filenames, image_dir):
    graph = nx.DiGraph()

    for i, j in gt_pairs:
        graph.add_edge(filenames[i], filenames[j])

    pos = nx.spring_layout(graph, seed=42)

    net = Network(height='1000px', width='100%', notebook=False, bgcolor='#222222', font_color='white')

    net.set_options("""
        {
          "physics": {
            "enabled": false
          },
          "edges": {
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 1
              }
            },
            "color": {
              "color": "lightgray"
            }
          }
        }
        """)

    for node, (x, y) in pos.items():
        filepath = os.path.join(image_dir, node)
        if os.path.exists(filepath):
            tooltip = image_to_data_url(filepath, size=150)
        else:
            tooltip = node

        net.add_node(
            node,
            label="",
            title=tooltip,
            x=x * 1000,
            y=y * 1000,
            shape="dot",
            size=10
        )

    for source, target in graph.edges():
        net.add_edge(source, target)

    net.show("static_graph_with_images.html", notebook=False)


def visualize_prediction_paths(prediction_tasks, stepwise_results, filenames, image_dir,
                               output_file="predictions_graph.html"):
    graph = nx.DiGraph()

    # Build graph from prediction tasks
    for (start, sequence), results in zip(prediction_tasks, stepwise_results):
        current = start
        for true_next, hit in zip(sequence, results):
            graph.add_edge(filenames[current], filenames[true_next], hit=hit)
            current = true_next

    pos = nx.spring_layout(graph, seed=42)

    net = Network(height='1000px', width='100%', notebook=False, bgcolor='#111111', font_color='white')

    net.set_options("""
    {
      "physics": {
        "enabled": false
      },
      "edges": {
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 1
          }
        }
      },
      "nodes": {
        "shape": "dot",
        "size": 10
      }
    }
    """)

    for node in graph.nodes():
        filepath = os.path.join(image_dir, node)
        tooltip = image_to_data_url(filepath, size=150) if os.path.exists(filepath) else node
        net.add_node(
            node,
            label="",
            title=tooltip,
            x=pos[node][0] * 1000,
            y=pos[node][1] * 1000,
        )

    for source, target, data in graph.edges(data=True):
        hit = data.get("hit", False)
        color = "green" if hit else "red"
        title = "Correct" if hit else "Incorrect"
        net.add_edge(source, target, color=color, title=title)

    net.show(output_file, notebook=False)


# === Main Pipeline ===
if __name__ == "__main__":
    embeddings, filenames = extract_clip_embeddings(IMAGE_DIR)
    training_pairs, prediction_tasks = split_sequences_for_training(GROUND_TRUTH_PATH, filenames)

    edge_index, edge_weight = build_weighted_hybrid_graph(
        embeddings, training_pairs, TOP_K_EDGES, GT_WEIGHT, SIM_WEIGHT
    )
    data = Data(x=embeddings, edge_index=edge_index, edge_attr=edge_weight)

    encoder = train_sequence_model_with_hard_negatives(
        data, training_pairs, HIDDEN_DIM, EPOCHS, TEMPERATURE, NUM_NEGATIVES
    )

    predictions = predict_next_in_sequence(data, encoder, filenames, training_pairs, TOP_K_PREDICTIONS)

    for img, next_imgs in predictions.items():
        if next_imgs:
            print(f"{img} --> {next_imgs[0:TOP_K_PREDICTIONS]}")

    stepwise_results = stepwise_sequence_prediction(
        data, encoder, filenames, prediction_tasks, TOP_K_PREDICTIONS
    )
    evaluate_stepwise_predictions(stepwise_results)

    visualize_graph_with_image_hover(training_pairs, filenames, IMAGE_DIR)
    visualize_prediction_paths(prediction_tasks, stepwise_results, filenames, IMAGE_DIR)

    print("Fertig. Modell trainiert und Vorhersagen abgeschlossen.")

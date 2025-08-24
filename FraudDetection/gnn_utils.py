#
# import torch
# import pandas as pd
# import matplotlib.pyplot as plt
# import networkx as nx
# from torch_geometric.data import HeteroData
# from torch_geometric.nn import GATConv
# from torch_geometric.utils import to_networkx
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
#
# def prepare_graph(df):
#     df = df[df["nameDest"].str.startswith("C") & df["nameOrig"].str.startswith("C")]
#     customers = pd.Index(pd.concat([df["nameOrig"], df["nameDest"]]).unique())
#     customer_id_map = {name: i for i, name in enumerate(customers)}
#
#     edge_index = torch.tensor([
#         df["nameOrig"].map(customer_id_map).values,
#         df["nameDest"].map(customer_id_map).values
#     ], dtype=torch.long)
#
#     edge_label = torch.tensor(df["isFraud"].values, dtype=torch.long)
#
#     data = HeteroData()
#     data["customer"].num_nodes = len(customers)
#     data["customer", "transfers", "customer"].edge_index = edge_index
#     data["customer", "transfers", "customer"].edge_label = edge_label
#     data["customer"].x = torch.eye(len(customers))
#
#     return data, edge_label, customer_id_map
#
# class GNN(torch.nn.Module):
#     def __init__(self, hidden_channels):
#         super().__init__()
#         self.conv1 = GATConv((-1, -1), hidden_channels)
#         self.conv2 = GATConv((-1, -1), 2)
#
#     def forward(self, x_dict, edge_index_dict):
#         x = self.conv1(x_dict['customer'], edge_index_dict[("customer", "transfers", "customer")])
#         x = x.relu()
#         x = self.conv2(x, edge_index_dict[("customer", "transfers", "customer")])
#         return x
#
# def train_gnn(data, edge_label, epochs=10):
#     model = GNN(hidden_channels=16)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
#     loss_fn = torch.nn.CrossEntropyLoss()
#
#     train_idx, test_idx = train_test_split(torch.arange(edge_label.size(0)), test_size=0.3, stratify=edge_label)
#
#     for epoch in range(epochs):
#         model.train()
#         optimizer.zero_grad()
#         out = model(data.x_dict, data.edge_index_dict)
#         loss = loss_fn(out[train_idx], edge_label[train_idx])
#         loss.backward()
#         optimizer.step()
#
#     model.eval()
#     with torch.no_grad():
#         out = model(data.x_dict, data.edge_index_dict)
#         preds = out[test_idx].argmax(dim=1)
#         report = classification_report(edge_label[test_idx], preds, output_dict=True)
#         return model, report
#
# def visualize_graph(data):
#     # Convert to NetworkX graph
#     G = to_networkx(data, edge_attrs=["edge_label"], to_undirected=True)
#
#     # Use edge_label inside data for coloring
#     edge_labels = data['customer', 'transfers', 'customer'].edge_label.tolist()
#     edge_colors = ['red' if label == 1 else 'green' for label in edge_labels]
#
#     # Draw graph
#     pos = nx.spring_layout(G, seed=42)
#     plt.figure(figsize=(10, 8))
#     nx.draw(
#         G, pos, with_labels=True,
#         edge_color=edge_colors,
#         node_color='skyblue',
#         node_size=400,
#         font_size=8
#     )
#     # Add legend
#     red_patch = plt.Line2D([0], [0], color='red', label='Fraud')
#     green_patch = plt.Line2D([0], [0], color='green', label='Legit')
#     plt.legend(handles=[red_patch, green_patch])
#     plt.title("Transaction Graph - Fraud (Red) vs Legit (Green)")
#     plt.show()
#

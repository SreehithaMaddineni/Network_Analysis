#  Network Analysis Project

This project focuses on **analyzing directed networks** using three core tasks:

1.  Influencer Detection – Find the most central or important person in a network  
2.  Link Prediction – Predict which new connections are likely to form  
3.  Community Detection – Discover groups or clusters of related nodes

---

##  Real-World Use Cases

- Social Media – Identify influencers, suggest friends, or detect user communities
- E-commerce – Recommend products by predicting co-purchases
- Cybersecurity – Detect clusters of suspicious activity in networks

---

##  What's Going On Behind the Scenes?

### 1. Influencer Detection – _Random Walk with Teleportation_

We simulate a "random surfer" who moves through the graph:

- Follows edges most of the time
- Occasionally "teleports" to a random node

The node visited the most is considered the "top influencer" — just like how Google's PageRank works!

---

### 2. Link Prediction – _Linear Combination (Least Squares)_

We try to guess missing links in the network by asking:

> “Can this row of the adjacency matrix be reconstructed using a combination of the others?”

This works well in social and biological networks where **structural patterns repeat**.

---

### 3. Community Detection – _Louvain Method_

We find communities by maximizing a value called "modularity", which checks:

- Are there more edges inside a group than between groups?

It converts the directed graph to undirected and finds densely connected clusters.

---

##  Project Structure

```text
 Network_Analysis_Project
├── Network_Analysis.py     ← Main code file
├── data.csv                ← Input CSV file (node connections)
├── requirements.txt        ← Dependencies
└── README.md               ← You are here!
```

---

##  Example Visualization

> When the script runs, it generates a colorful plot showing how your nodes are grouped by community.

---

## How to Run

###  Install dependencies

```bash
pip install -r requirements.txt
```

###  Run the project

```bash
python Network_Analysis.py
```

---

##  Input Format (CSV)

Your `data.csv` file should look like this:

```csv
Source, Target1, Target2, ...
A, B, C
B, C
C, A
```

Each row represents one node and the people they connect to.

---

##  Limitations

- Works best on **medium-sized graphs** (a few hundred nodes)
- Assumes the input file is **clean and consistently formatted**
- Link prediction is **symmetric** in behavior due to matrix manipulation, even though the graph is directed

---

##  License

This project is open-source under the **MIT License**.

---

##  Author

**Maddineni Sreehitha**  
_3rd Year B.Tech – Mathematics and Computing_  
> “Built this to explore graph theory and network science hands-on!”

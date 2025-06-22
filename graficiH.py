import matplotlib.pyplot as plt
import numpy as np

# Dati
fastica_scores = [-0.38, -0.30, -0.26, -0.18, -0.16]
infomax_scores = [-0.21, -0.19, -0.19, -0.17, -0.16]
tests = ['Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5']
x = np.arange(len(tests))
width = 0.35

# --- Grafico a Barre Raggruppate (FastICA a sinistra, Infomax a destra) ---
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, fastica_scores, width, label='FastICA', color='#0077FF')  # blu elettrico
bars2 = ax.bar(x + width/2, infomax_scores, width, label='Extended Infomax', color='#FF3B3F')  # rosso acceso

# Etichette
ax.set_xlabel('Test')
ax.set_ylabel('Silhouette Score')
ax.set_title('Performance del t-SNE riguardo la silhouette score in base all\'algoritmo')
ax.set_xticks(x)
ax.set_xticklabels(tests)
ax.set_ylim(-0.4, -0.1)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.6)

# Etichette sopra le barre
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # spostamento verticale
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# --- Grafico a Linee (con colori accesi) ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(tests, fastica_scores, marker='o', linestyle='-', color='#0077FF', label='FastICA')
ax.plot(tests, infomax_scores, marker='o', linestyle='-', color='#FF3B3F', label='Extended Infomax')

ax.set_xlabel('Test')
ax.set_ylabel('Silhouette Score')
ax.set_title('Performance del t-SNE riguardo la silhouette score in base all\'algoritmo')
ax.set_ylim(-0.4, -0.1)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

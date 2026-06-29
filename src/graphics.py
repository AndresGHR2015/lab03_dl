import json
import matplotlib.pyplot as plt
import numpy as np

modelos = ["mlp", "cnn", "resnet", "ResNet_FineTuned", "CNN_Ablacion"]
nombres_legibles = ["E2: MLP Multitarea", "E3: CNN Simple", "E4: ResNet (Congelada)", "E5: ResNet (Fine-Tuning)", "E6: Ablación (λ=0.5)"]
genero_acc = []
edad_mae = []

for m in modelos:
    try:
        with open(f"outputs/models/{m}_metrics.json", "r") as f:
            data = json.load(f)
            genero_acc.append(data.get("gender_accuracy", 0.0))
            edad_mae.append(data.get("age_mae", 0.0))
    except FileNotFoundError:
        print(f"Advertencia: No se encontró el JSON para {m}")
        genero_acc.append(0.0)
        edad_mae.append(0.0)

nombres_legibles.insert(0, "E1: Baseline (PCA + GNB/Ridge)")
genero_acc.insert(0, 0.8128)  
edad_mae.insert(0, 11.7690) 

fig, ax1 = plt.subplots(figsize=(12, 6))

color = '#e74c3c'
ax1.set_xlabel('Experimentos del Laboratorio', fontweight='bold', labelpad=12)
ax1.set_ylabel('Edad - Error Absoluto Medio (MAE en años)', color=color, fontweight='bold')
bars = ax1.bar(np.arange(len(nombres_legibles)) - 0.2, edad_mae, width=0.4, color=color, alpha=0.7, label='MAE Edad (Menor es mejor)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(np.arange(len(nombres_legibles)))
ax1.set_xticklabels(nombres_legibles, rotation=15, ha="right")
ax2 = ax1.twinx()
color = '#2c3e50'
ax2.set_ylabel('Género - Exactitud (Accuracy)', color=color, fontweight='bold')
line = ax2.plot(np.arange(len(nombres_legibles)) + 0.2, genero_acc, color=color, marker='o', linewidth=2.5, markersize=8, label='Acc Género (Mayor es mejor)')
ax2.tick_params(axis='y', labelcolor=color)
plt.title('Comparativa Global de Rendimiento: Modelos Clásicos vs Deep Learning (UTKFace)', fontsize=14, fontweight='bold', pad=20)
fig.tight_layout()
plt.savefig('comparativa_modelos_utkface.png', dpi=300)
print("¡Gráfico 'comparativa_modelos_utkface.png' generado con éxito!")
plt.show()
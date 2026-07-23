import matplotlib.pyplot as plt
import numpy as np

# Accuracy values from Phase 5
models = [
    "Random Forest",
    "Neural Network",
    "Logistic Regression",
    "KNN"
]

accuracy = [
    99.78,
    99.03,
    98.76,
    73.78
]

# Create Bar Chart
plt.figure(figsize=(8, 5))

plt.bar(models, accuracy)

plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy (%)")
plt.ylim(60, 100)

# Display values on top of bars
for i, value in enumerate(accuracy):
    plt.text(i, value + 0.3, f"{value:.2f}%", ha="center")

plt.tight_layout()

# Save graph
plt.savefig("../output/model_accuracy.png")

print("Graph saved successfully!")

# Display graph
plt.show()
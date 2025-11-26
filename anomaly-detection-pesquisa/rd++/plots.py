import matplotlib.pyplot as plt

# Insert AUROC and latency values here
# Average Pixel AUROC values from the table
# Example structure: {"Technique": ([auroc_values], [latency_values])}
data = {
    "DifferNet": ([55.092], [13.36]),
    "DifferNet + CBAM": ([57.332], [15.39]),
    "DifferNet + SENET": ([60.114], [13.50]),
    "RD++": ([91.774], [8.19]),
    "RD++ + CBAM": ([91.434], [7.06]),
    "RD++ + SENET": ([92.522], [6.46]),
    "SimpleNet": ([65.14], [8.35]),
    "SimpleNet + CBAM": ([67.58], [16.80]),
    "SimpleNet + SENET": ([63.21], [11.78]),
    "EfficientAD": ([46.66], [2.51]),
    "EfficientAD + CBAM": ([45.25], [4.33]),
    "EfficientAD + SENET": ([43.00], [5.03])
}

# Assign unique colors to each main technique
colors = {
    "DifferNet": "blue",
    "RD++": "green",
    "SimpleNet": "red",
    "EfficientAD": "purple"
}

# Assign acronyms for each technique
acronyms = {
    "DifferNet": "DF",
    "DifferNet + CBAM": "DF-CBAM",
    "DifferNet + SENET": "DF-SENET",
    "RD++": "RD",
    "RD++ + CBAM": "RD-CBAM",
    "RD++ + SENET": "RD-SENET",
    "SimpleNet": "SN",
    "SimpleNet + CBAM": "SN-CBAM",
    "SimpleNet + SENET": "SN-SENET",
    "EfficientAD": "EA",
    "EfficientAD + CBAM": "EA-CBAM",
    "EfficientAD + SENET": "EA-SENET"
}

# Initialize the plot
plt.figure(figsize=(10, 6))

# Generate scatter points for each technique and its variants
for technique, (auroc_values, latency_values) in data.items():
    main_technique = technique.split(" ")[0].strip()
    color = colors.get(main_technique, "black")  # Default to black if not found
    plt.scatter(latency_values, auroc_values, label=technique, color=color)
    # Add acronym text near each point
    for x, y in zip(latency_values, auroc_values):
        plt.text(x, y, acronyms[technique], fontsize=8, ha='right')

# Customize the plot
plt.title("Pixel AUROC vs Latency (ms)")
plt.xlabel("Latency (ms)")
plt.ylabel("Pixel AUROC")
plt.legend(title="Techniques", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()

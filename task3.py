import csv

# Define the file paths
input_file = "manualAnnotationTask/Italian.label"  # Replace with your actual file path
output_file = "Refined_Italian.label"  # The refined output file

# Define rules for refining
false_positives = ["light rail", "great toppings", "get a whole", "with friendly service", "market and"]
false_negatives = {"carbonara": 1, "gelato": 1, "tiramisu": 1}  # Add missing dishes with correct labels

# New phrases to add (if needed)
new_phrases = ["panna cotta", "fettuccine alfredo"]

def refine_labels(input_file, output_file, false_positives, false_negatives, new_phrases):
    refined_lines = []
    existing_phrases = set()

    # Read the input file
    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            phrase, label = line.strip().split("\t")
            label = int(label)

            # Remove false positives
            if phrase in false_positives:
                continue

            # Fix false negatives
            if phrase in false_negatives:
                label = false_negatives[phrase]

            # Append the refined line
            refined_lines.append(f"{phrase}\t{label}")
            existing_phrases.add(phrase)

    # Add new phrases with positive labels if not already in the file
    for phrase in new_phrases:
        if phrase not in existing_phrases:
            refined_lines.append(f"{phrase}\t1")

    # Write the refined lines to the output file
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("\n".join(refined_lines))

# Execute the function
refine_labels(input_file, output_file, false_positives, false_negatives, new_phrases)

print(f"Refined file saved as {output_file}")

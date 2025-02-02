import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

input_sentence = "Hey, can you suggest a good movie for the weekend?"

responses = {
    "DialoGPT": "Sure! How about watching Inception? It's a great mix of action and mind-bending storytelling.",
    "BlenderBot": "I’d recommend The Grand Budapest Hotel! It’s a visually stunning and fun movie for a weekend watch.",
    "GPT-3.5": "Of course! Are you in the mood for action, comedy, or something thought-provoking? Interstellar is a great choice if you enjoy sci-fi.",
    "GPT-4": "I’d go with The Dark Knight if you love action or Parasite if you enjoy thrillers!",
    "T5": "You might like Knives Out! It's a mystery thriller with a fantastic cast.",
    "LLaMA-2": "I’d say Parasite is a great weekend pick—thrilling and unexpected!",
    "Mistral-7B": "I recommend Dune if you enjoy sci-fi epics with stunning visuals.",
    "Gemini-1.5": "For a great weekend movie, try The Matrix—it’s action-packed and thought-provoking.",
    "Claude-2": "I think Everything Everywhere All at Once is a fantastic choice!"
}

decision_matrix = np.array([
    [18, 0.30, 0.25, 0.60, 4.2, 3.8, 250, 1.2],  # DialoGPT
    [15, 0.35, 0.28, 0.62, 4.5, 4.0, 300, 1.5],  # BlenderBot
    [12, 0.40, 0.30, 0.65, 4.8, 4.5, 150, 3.0],  # GPT-3.5
    [10, 0.45, 0.32, 0.68, 5.0, 4.7, 140, 3.5],  # GPT-4
    [20, 0.28, 0.22, 0.58, 3.9, 3.5, 200, 1.0],  # T5
    [14, 0.38, 0.27, 0.63, 4.6, 4.2, 180, 1.8],  # LLaMA-2
    [13, 0.42, 0.29, 0.66, 4.9, 4.6, 160, 2.5],  # Mistral-7B
    [11, 0.44, 0.31, 0.67, 4.95, 4.65, 130, 3.2],# Gemini-1.5
    [12, 0.41, 0.30, 0.64, 4.85, 4.55, 145, 2.8] # Claude-2
])

models = list(responses.keys())

norm_matrix = decision_matrix / np.sqrt((decision_matrix**2).sum(axis=0))

weights = np.array([0.15, 0.15, 0.10, 0.15, 0.15, 0.10, 0.10, 0.10])  

weighted_matrix = norm_matrix * weights

ideal_best = np.max(weighted_matrix, axis=0) 
ideal_worst = np.min(weighted_matrix, axis=0)

dist_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
dist_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))

topsis_scores = dist_worst / (dist_best + dist_worst)

ranking_indices = np.argsort(topsis_scores)[::-1] 

df_results = pd.DataFrame({
    "Model": np.array(models)[ranking_indices],
    "TOPSIS Score": topsis_scores[ranking_indices]
})

print("\nTOPSIS Ranking for Conversational AI Models:")
print(df_results.to_string())

df_results.to_csv("topsis_ranking.csv", index=False)

df_responses = pd.DataFrame(list(responses.items()), columns=["Model", "Response"])
print("\nModel Responses:")
print(df_responses.to_string())

plt.figure(figsize=(10, 5))
plt.bar(df_results["Model"], df_results["TOPSIS Score"], color='lightcoral')
plt.title("TOPSIS Scores for Conversational AI Models")
plt.xlabel("Model")
plt.ylabel("TOPSIS Score")
plt.xticks(rotation=45)
plt.show()
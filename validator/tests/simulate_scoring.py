from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
import numpy as np
import secrets

# Updated constants to match scoring_utils.py
W_PREDICTION = 1.0
W_COMPUTER_VISION = 1.1
W_DIVERSITY_BONUS = 1.3  # 15% bonus for doing both types

def simulate_miners(
    num_miners: int,
    num_tasks: int,
    prediction_ratio: float = 0.4,
    both_tasks_ratio: float = 0.3
) -> pd.DataFrame:
    """Simulate miner performance using actual scoring logic."""
    
    results = []
    
    # Assign realistic skill levels and task preferences to miners
    miner_skills = {}
    for i in range(num_miners):
        # Randomly assign miner type based on both_tasks_ratio
        r = secrets.SystemRandom().random()
        if r < both_tasks_ratio:
            task_types = {"prediction", "cv"}
        elif r < (1 + both_tasks_ratio) / 2:
            task_types = {"prediction"}
        else:
            task_types = {"cv"}
            
        miner_skills[f"miner_{i}"] = {
            # Even tighter ranges for more realistic distribution
            "prediction_accuracy": secrets.SystemRandom().uniform(0.5, 0.6),
            "confidence_range": (0.2, 0.8),
            "processing_speed": secrets.SystemRandom().uniform(0.95, 1.0),
            "cv_quality": secrets.SystemRandom().uniform(0.85, 0.95),
            "task_types": task_types
        }
    
    # Get baseline processing times
    avg_prediction_time = 5.0  # seconds
    avg_cv_time = 30.0  # seconds
    
    for task_idx in range(num_tasks):
        for miner_id, skills in miner_skills.items():
            # Determine task type
            is_prediction = secrets.SystemRandom().random() < prediction_ratio
            task_type = "prediction" if is_prediction else "cv"
            
            # Skip if miner doesn't do this type of task
            if task_type not in skills["task_types"]:
                continue
            
            if is_prediction:
                # Simulate prediction task - increase base scores
                is_correct = secrets.SystemRandom().random() < skills["prediction_accuracy"]
                confidence = secrets.SystemRandom().uniform(*skills["confidence_range"])
                processing_time = avg_prediction_time * (1/skills["processing_speed"])
                
                if is_correct:
                    base_score = 0.8 + (0.5 * confidence)  # Higher base score for correct predictions
                else:
                    base_score = 0.4 * (1 - confidence)  # Less severe penalty for wrong predictions
                
                score = base_score * W_PREDICTION
                
            else:
                # Modified CV scoring to be more in line with predictions
                completeness = skills["cv_quality"]
                processing_time = avg_cv_time * (1/skills["processing_speed"])
                time_factor = secrets.SystemRandom().uniform(0.85, 1.0)
                
                # Adjusted CV scoring formula to match prediction score ranges
                base_score = (0.8 * completeness + 0.2 * time_factor) * 0.8  # Scale down base score
                score = base_score * W_COMPUTER_VISION
            
            results.append({
                "task_id": task_idx,
                "miner_id": miner_id,
                "task_type": task_type,
                "score": score,
                "processing_time": processing_time
            })
    
    df = pd.DataFrame(results)
    
    # Calculate aggregate scores and task participation per miner
    miner_stats = df.groupby("miner_id").agg({
        "score": "mean",
        "processing_time": "mean",
        "task_type": lambda x: set(x)
    }).reset_index()
    
    # Add diversity bonus for miners doing both tasks
    miner_stats['does_both'] = miner_stats['task_type'].apply(lambda x: len(x) > 1)
    miner_stats['score'] = miner_stats.apply(
        lambda row: row['score'] * W_DIVERSITY_BONUS if row['does_both'] else row['score'],
        axis=1
    )
    
    return miner_stats

# Run simulation with more miners
num_miners = 50
num_tasks = 1000

df = simulate_miners(
    num_miners=num_miners,
    num_tasks=num_tasks,
    prediction_ratio=0.4,
    both_tasks_ratio=0.3
)

# Sort miners by score
df = df.sort_values('score', ascending=True)

# Create bar plot
plt.figure(figsize=(15, 8))

# Color bars based on task participation
colors = []
for task_types in df['task_type']:
    if len(task_types) > 1:
        colors.append('red')  # Both tasks
    elif 'prediction' in task_types:
        colors.append('skyblue')  # Prediction only
    else:
        colors.append('green')  # CV only

bars = plt.bar(range(len(df)), df["score"], width=0.8, color=colors)

plt.title("Miner Incentive Scores (Sorted by Score)")
plt.xlabel("Miner Rank")
plt.ylabel("Calculated Incentive Score")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', label='Prediction + CV Tasks'),
    Patch(facecolor='skyblue', label='Prediction Only'),
    Patch(facecolor='green', label='CV Only')
]
plt.legend(handles=legend_elements)

# Add miner IDs as x-tick labels
plt.xticks(range(len(df)), df["miner_id"], rotation=45, ha='right')

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom',
             fontsize=8)

plt.tight_layout()
plt.show()

# Print summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Print task type distribution
print("\nTask Type Distribution:")
print(f"Miners doing both tasks: {sum(len(t) > 1 for t in df['task_type'])}")
print(f"Miners doing predictions only: {sum(len(t) == 1 and 'prediction' in t for t in df['task_type'])}")
print(f"Miners doing CV only: {sum(len(t) == 1 and 'cv' in t for t in df['task_type'])}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Set style for scientific publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

def analyze_and_visualize():
    # 1. Load Data
    df = pd.read_csv('strategy_dataset.csv')
    print("Data loaded successfully.")
    
    # 2. Strategy Comparison (The "Why LAND?" Plot)
    # We need to melt the dataframe to plot side-by-side boxplots
    df_scores = df[['Score_RTH', 'Score_HOVER', 'Score_LAND']].melt(var_name='Strategy', value_name='Dispersion_Score')
    
    # Clean up names for plotting
    df_scores['Strategy'] = df_scores['Strategy'].str.replace('Score_', '')
    
    plt.figure(figsize=(10, 6))
    # Use log scale because RTH variance might be orders of magnitude larger
    sns.boxplot(x='Strategy', y='Dispersion_Score', data=df_scores, palette="Set2")
    plt.yscale('log')
    plt.title('Comparison of Position Uncertainty (Dispersion) by Strategy', fontsize=14)
    plt.ylabel('Dispersion Score (Log Scale)\n(Lower is Better)', fontsize=12)
    plt.xlabel('Strategy', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig('strategy_comparison_boxplot.png', dpi=300)
    print("Saved 'strategy_comparison_boxplot.png'")

    # 3. Decision Tree Training
    # Features
    X = df[['Wind_Speed', 'Dist_to_Origin', 'Altitude']]
    y = df['Best_Strategy']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    print("\nModel Accuracy:", clf.score(X_test, y_test))
    print("\nClassification Report:")
    # Handle case where not all classes are in test set
    unique_labels = y.unique()
    print(classification_report(y_test, clf.predict(X_test), labels=unique_labels, zero_division=0))
    
    # 4. Visualize Decision Tree
    plt.figure(figsize=(12, 8))
    plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True, rounded=True, fontsize=10)
    plt.title('Decision Tree for Strategy Selection', fontsize=14)
    plt.tight_layout()
    plt.savefig('decision_tree_viz.png', dpi=300)
    print("Saved 'decision_tree_viz.png'")
    
    # 5. Feature Analysis: Wind Speed vs LAND Score
    # Show how wind affects the "best" strategy's performance
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Wind_Speed', y='Score_LAND', alpha=0.6, color='green')
    plt.title('Impact of Wind Speed on LAND Strategy Uncertainty', fontsize=14)
    plt.xlabel('Wind Speed (m/s)', fontsize=12)
    plt.ylabel('LAND Dispersion Score', fontsize=12)
    plt.tight_layout()
    plt.savefig('wind_vs_land_score.png', dpi=300)
    print("Saved 'wind_vs_land_score.png'")

    # 6. Analyze the outlier (HOVER case)
    hover_cases = df[df['Best_Strategy'] == 'HOVER']
    if not hover_cases.empty:
        print("\nAnalyzing HOVER cases:")
        print(hover_cases)
    else:
        print("\nNo HOVER cases found in dataset.")

if __name__ == "__main__":
    analyze_and_visualize()

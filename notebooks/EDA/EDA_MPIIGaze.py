import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime

class MPIIGazeAnalyzer:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.csv_path = os.path.join(base_dir, "outputs", "MPIIGAZE", "mpiigaze_features.csv")
        self.output_dir = os.path.join(base_dir, "outputs", "analysis")
        os.makedirs(self.output_dir, exist_ok=True)

        plt.style.use('default')
        sns.set_palette("husl")

    def load_and_analyze(self):
        print("📊 Loading MPIIGAZE dataset...")

        df = pd.read_csv(self.csv_path)

        print(f"✅ Dataset loaded: {len(df):,} samples")
        print(f"📁 Subjects: {df['subject'].nunique()}")
        print(f"📅 Days per subject: {df.groupby('subject')['day'].nunique().to_dict()}")

        return df

    def create_comprehensive_analysis(self, df):
        print("📈 Creating comprehensive analysis...")

        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle("MPIIGAZE Dataset - Comprehensive Analysis", fontsize=18, fontweight='bold')

        # 1. EAR Distribution
        axes[0,0].hist(df['avg_ear'], bins=50, alpha=0.8, edgecolor='black')
        axes[0,0].axvline(df['avg_ear'].mean(), color='red', linestyle='--', label=f"Mean: {df['avg_ear'].mean():.3f}")
        axes[0,0].set_title("EAR Distribution")
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # 2. Attention Distribution
        axes[0,1].hist(df['attention_score'], bins=50, alpha=0.8, edgecolor='black', color='limegreen')
        axes[0,1].axvline(df['attention_score'].mean(), color='red', linestyle='--', label=f"Mean: {df['attention_score'].mean():.3f}")
        axes[0,1].set_title("Attention Score Distribution")
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        # 3. Gaze Velocity Distribution
        axes[0,2].hist(df['gaze_velocity'], bins=50, alpha=0.8, edgecolor='black', color='orange')
        axes[0,2].axvline(df['gaze_velocity'].mean(), color='red', linestyle='--', label=f"Mean: {df['gaze_velocity'].mean():.3f}")
        axes[0,2].set_title("Gaze Velocity Distribution")
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)

        # 4. Subject-wise metrics
        subject_stats = df.groupby('subject').agg({
            'attention_score': 'mean',
            'avg_ear': 'mean',
            'gaze_velocity': 'mean',
            'image': 'count'
        }).rename(columns={'image': 'sample_count'})

        subject_stats[['attention_score', 'avg_ear', 'gaze_velocity']].plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title("Subject-wise Cognitive Metrics")
        axes[1,0].set_xlabel("Subject")
        axes[1,0].grid(True, alpha=0.3)

        # 5. EAR vs Attention
        sc = axes[1,1].scatter(df['avg_ear'], df['attention_score'], 
                               c=df['gaze_velocity'], cmap='viridis', alpha=0.6, s=5)
        axes[1,1].set_title("EAR vs Attention (Color = Gaze Velocity)")
        axes[1,1].set_xlabel("EAR")
        axes[1,1].set_ylabel("Attention")
        plt.colorbar(sc, ax=axes[1,1])

        # 6. Gaze Scatter
        axes[1,2].scatter(df['gaze_x'], df['gaze_y'], alpha=0.3, s=2)
        axes[1,2].axhline(0, color='red', linestyle='--')
        axes[1,2].axvline(0, color='red', linestyle='--')
        axes[1,2].set_title("Gaze Direction Distribution")
        axes[1,2].set_xlabel("Gaze X")
        axes[1,2].set_ylabel("Gaze Y")

        # 7. Correlation Heatmap
        numeric_cols = [
            'avg_ear', 'left_ear', 'right_ear', 'gaze_x', 'gaze_y',
            'gaze_magnitude', 'attention_score', 'gaze_velocity', 'gaze_entropy'
        ]
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, ax=axes[2,0], cmap='coolwarm', annot=True, fmt=".2f")
        axes[2,0].set_title("Feature Correlation Matrix")

        # 8. Temporal attention pattern
        sample = df[(df['subject'] == 'p00') & (df['day'] == 'day01')].head(1000)
        axes[2,1].plot(sample['attention_score'], label='Attention')
        axes[2,1].plot(sample['avg_ear'], label='EAR', alpha=0.7)
        axes[2,1].set_title("Temporal Pattern (p00/day01)")
        axes[2,1].legend()
        axes[2,1].grid(True)

        # 9. Eye State Distribution
        eye_counts = df['eyes_open'].value_counts()
        labels = ['Closed', 'Open'] if len(eye_counts) == 2 else eye_counts.index.astype(str)
        eye_counts.plot(kind='pie', ax=axes[2,2], autopct='%1.1f%%', colors=['salmon', 'lightgreen'])
        axes[2,2].set_title("Eye State Distribution")

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "mpiigaze_eda.png")
        plt.savefig(plot_path, dpi=300)
        plt.show()

        print(f"📸 EDA plot saved to: {plot_path}")

    def generate_statistical_report(self, df):
        print("📋 Generating statistical report...")

        report = {
            "analysis_date": datetime.now().isoformat(),
            "total_samples": len(df),
            "subjects": df['subject'].unique().tolist(),
            "mean_attention": float(df['attention_score'].mean()),
            "std_attention": float(df['attention_score'].std()),
            "high_attention_ratio": float((df['attention_score'] > 0.7).mean()),
            "low_attention_ratio": float((df['attention_score'] < 0.3).mean()),
            "eyes_open_ratio": float(df['eyes_open'].mean()),
            "avg_ear_mean": float(df['avg_ear'].mean()),
            "avg_ear_std": float(df['avg_ear'].std())
        }

        report_path = os.path.join(self.output_dir, "mpiigaze_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print("✅ REPORT SUMMARY")
        print(json.dumps(report, indent=2))
        print(f"📄 Report saved to: {report_path}")


def main():
    base_dir = r"C:\Users\Shail\Downloads\neo_cognition"
    analyzer = MPIIGazeAnalyzer(base_dir)

    df = analyzer.load_and_analyze()
    analyzer.create_comprehensive_analysis(df)
    analyzer.generate_statistical_report(df)


if __name__ == "__main__":
    main()
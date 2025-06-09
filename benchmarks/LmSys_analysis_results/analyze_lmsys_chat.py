from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from typing import Dict, List, Tuple
import json

def load_lmsys_data():
    """
    Load LMSYS Chat dataset from Hugging Face
    """
    try:
        # Load the dataset
        dataset = load_dataset("lmsys/lmsys-chat-1m")
        print("Dataset structure:", dataset)
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def convert_numpy_types(obj):
    """
    Convert numpy types to native Python types for JSON serialization
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def analyze_conversation_lengths(df: pd.DataFrame) -> Dict:
    """
    Analyze conversation lengths and patterns
    """
    # Calculate conversation lengths
    df['conversation_length'] = df['conversation'].apply(len)
    
    # Basic statistics
    stats = {
        'total_conversations': int(len(df)),
        'avg_conversation_length': float(df['conversation_length'].mean()),
        'median_conversation_length': float(df['conversation_length'].median()),
        'min_conversation_length': int(df['conversation_length'].min()),
        'max_conversation_length': int(df['conversation_length'].max()),
    }
    
    # Plot distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='conversation_length', bins=50)
    plt.title('Distribution of Conversation Lengths')
    plt.xlabel('Number of Messages')
    plt.ylabel('Count')
    plt.savefig('conversation_lengths.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return stats

def analyze_model_usage(df: pd.DataFrame) -> Dict:
    """
    Analyze model usage patterns - models are in the 'model' column
    """
    # Count models from the model column
    model_counts = df['model'].value_counts()
    print(f"Found {len(model_counts)} unique models")
    print("Top 10 models:")
    print(model_counts.head(10))
    
    # Plot model usage (top 20 models)
    plt.figure(figsize=(15, 8))
    top_models = model_counts.head(20)
    
    sns.barplot(x=top_models.values, y=top_models.index)
    plt.title('Top 20 Model Usage Distribution')
    plt.xlabel('Number of Conversations')
    plt.tight_layout()
    plt.savefig('model_usage.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Convert to regular dict with Python int values
    return {str(k): int(v) for k, v in model_counts.items()}

def analyze_message_types(df: pd.DataFrame) -> Dict:
    """
    Analyze message types and roles
    """
    role_counts = Counter()
    for conv in df['conversation']:
        for msg in conv:
            if 'role' in msg:
                role_counts[msg['role']] += 1
    
    # Plot role distribution
    plt.figure(figsize=(10, 6))
    if role_counts:
        roles_df = pd.DataFrame.from_dict(role_counts, orient='index', columns=['count'])
        sns.barplot(x=roles_df.index, y=roles_df['count'])
        plt.title('Message Role Distribution')
        plt.tight_layout()
        plt.savefig('message_roles.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return {k: int(v) for k, v in role_counts.items()}

def analyze_language_distribution(df: pd.DataFrame) -> Dict:
    """
    Analyze language distribution in the dataset
    """
    language_counts = df['language'].value_counts()
    print(f"Found {len(language_counts)} unique languages")
    print("Top 10 languages:")
    print(language_counts.head(10))
    
    # Plot language distribution (top 15)
    plt.figure(figsize=(12, 8))
    top_languages = language_counts.head(15)
    
    sns.barplot(x=top_languages.values, y=top_languages.index)
    plt.title('Top 15 Language Distribution')
    plt.xlabel('Number of Conversations')
    plt.tight_layout()
    plt.savefig('language_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {str(k): int(v) for k, v in language_counts.items()}

def analyze_turn_patterns(df: pd.DataFrame) -> Dict:
    """
    Analyze conversation turn patterns
    """
    turn_counts = df['turn'].value_counts().sort_index()
    
    # Plot turn distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(x=turn_counts.index, y=turn_counts.values)
    plt.title('Distribution of Conversation Turns')
    plt.xlabel('Turn Number')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('turn_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'avg_turns': float(df['turn'].mean()),
        'max_turns': int(df['turn'].max()),
        'min_turns': int(df['turn'].min()),
        'turn_distribution': {int(k): int(v) for k, v in turn_counts.items()}
    }

def analyze_content_patterns(df: pd.DataFrame) -> Dict:
    """
    Analyze content patterns in messages
    """
    user_message_lengths = []
    assistant_message_lengths = []
    
    # Calculate message lengths by role
    for conv in df['conversation']:
        for msg in conv:
            if 'content' in msg and 'role' in msg:
                length = len(msg['content'].split())
                if msg['role'] == 'user':
                    user_message_lengths.append(length)
                elif msg['role'] == 'assistant':
                    assistant_message_lengths.append(length)
    
    # Plot message length distributions
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    if user_message_lengths:
        sns.histplot(user_message_lengths, bins=50, alpha=0.7)
        plt.title('User Message Length Distribution')
        plt.xlabel('Words')
        plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    if assistant_message_lengths:
        sns.histplot(assistant_message_lengths, bins=50, alpha=0.7)
        plt.title('Assistant Message Length Distribution')
        plt.xlabel('Words')
        plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('message_lengths.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    content_stats = {}
    if user_message_lengths:
        content_stats['user_messages'] = {
            'avg_length': float(np.mean(user_message_lengths)),
            'max_length': int(max(user_message_lengths)),
            'min_length': int(min(user_message_lengths)),
            'median_length': float(np.median(user_message_lengths))
        }
    
    if assistant_message_lengths:
        content_stats['assistant_messages'] = {
            'avg_length': float(np.mean(assistant_message_lengths)),
            'max_length': int(max(assistant_message_lengths)),
            'min_length': int(min(assistant_message_lengths)),
            'median_length': float(np.median(assistant_message_lengths))
        }
    
    return content_stats

def analyze_moderation_patterns(df: pd.DataFrame) -> Dict:
    """
    Analyze OpenAI moderation patterns
    """
    if 'openai_moderation' in df.columns:
        # Count moderation flags
        moderation_stats = {}
        
        # Check if any conversations were flagged
        flagged_count = 0
        for moderation in df['openai_moderation']:
            if moderation and any(moderation.get(key, False) for key in moderation if key != 'flagged'):
                flagged_count += 1
        
        moderation_stats = {
            'total_conversations': int(len(df)),
            'flagged_conversations': int(flagged_count),
            'flagged_percentage': float(flagged_count / len(df) * 100)
        }
        
        return moderation_stats
    return {}

def create_summary_report(analyses: Dict) -> str:
    """
    Create a summary report of the analysis
    """
    report = []
    report.append("="*60)
    report.append("LMSYS Chat 1M Dataset Analysis Summary")
    report.append("="*60)
    
    # Conversation stats
    if 'conversation_lengths' in analyses:
        stats = analyses['conversation_lengths']
        report.append(f"\n📊 Conversation Statistics:")
        report.append(f"   Total conversations: {stats['total_conversations']:,}")
        report.append(f"   Average length: {stats['avg_conversation_length']:.2f} messages")
        report.append(f"   Median length: {stats['median_conversation_length']:.0f} messages")
        report.append(f"   Range: {stats['min_conversation_length']}-{stats['max_conversation_length']} messages")
    
    # Model stats
    if 'model_usage' in analyses:
        models = analyses['model_usage']
        report.append(f"\n🤖 Model Statistics:")
        report.append(f"   Unique models: {len(models)}")
        if models:
            top_model = max(models.items(), key=lambda x: x[1])
            report.append(f"   Most used model: {top_model[0]} ({top_model[1]:,} conversations)")
    
    # Language stats
    if 'language_distribution' in analyses:
        languages = analyses['language_distribution']
        report.append(f"\n🌍 Language Statistics:")
        report.append(f"   Unique languages: {len(languages)}")
        if languages:
            top_lang = max(languages.items(), key=lambda x: x[1])
            report.append(f"   Most common language: {top_lang[0]} ({top_lang[1]:,} conversations)")
    
    # Content stats
    if 'content_patterns' in analyses:
        content = analyses['content_patterns']
        report.append(f"\n💬 Content Statistics:")
        if 'user_messages' in content:
            user_stats = content['user_messages']
            report.append(f"   User message avg length: {user_stats['avg_length']:.1f} words")
        if 'assistant_messages' in content:
            asst_stats = content['assistant_messages']
            report.append(f"   Assistant message avg length: {asst_stats['avg_length']:.1f} words")
    
    return "\n".join(report)

def main():
    # Load dataset
    dataset = load_lmsys_data()
    if dataset is None:
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(dataset['train'])
    
    print(f"\nDataset loaded: {len(df):,} conversations")
    print(f"Columns: {list(df.columns)}")
    
    # Debug: Print first conversation structure
    print("\nFirst conversation structure:")
    if len(df) > 0:
        first_conv = df['conversation'].iloc[0]
        print(json.dumps(first_conv, indent=2))
    
    # Perform analyses
    print("\n🔍 Starting analysis...")
    analyses = {}
    
    try:
        analyses['conversation_lengths'] = analyze_conversation_lengths(df)
        print("✅ Conversation length analysis complete")
    except Exception as e:
        print(f"❌ Error in conversation length analysis: {e}")
    
    try:
        analyses['model_usage'] = analyze_model_usage(df)
        print("✅ Model usage analysis complete")
    except Exception as e:
        print(f"❌ Error in model usage analysis: {e}")
    
    try:
        analyses['message_types'] = analyze_message_types(df)
        print("✅ Message type analysis complete")
    except Exception as e:
        print(f"❌ Error in message type analysis: {e}")
    
    try:
        analyses['language_distribution'] = analyze_language_distribution(df)
        print("✅ Language distribution analysis complete")
    except Exception as e:
        print(f"❌ Error in language distribution analysis: {e}")
    
    try:
        analyses['turn_patterns'] = analyze_turn_patterns(df)
        print("✅ Turn pattern analysis complete")
    except Exception as e:
        print(f"❌ Error in turn pattern analysis: {e}")
    
    try:
        analyses['content_patterns'] = analyze_content_patterns(df)
        print("✅ Content pattern analysis complete")
    except Exception as e:
        print(f"❌ Error in content pattern analysis: {e}")
    
    try:
        analyses['moderation_patterns'] = analyze_moderation_patterns(df)
        print("✅ Moderation pattern analysis complete")
    except Exception as e:
        print(f"❌ Error in moderation pattern analysis: {e}")
    
    # Convert numpy types for JSON serialization
    analyses = convert_numpy_types(analyses)
    
    # Save analysis results
    try:
        with open('lmsys_analysis_results.json', 'w') as f:
            json.dump(analyses, f, indent=2)
        print("✅ Results saved to 'lmsys_analysis_results.json'")
    except Exception as e:
        print(f"❌ Error saving results: {e}")
    
    # Create and display summary report
    summary = create_summary_report(analyses)
    print(summary)
    
    # Save summary report
    with open('lmsys_summary_report.txt', 'w') as f:
        f.write(summary)
    
    print("\n🎉 Analysis complete!")
    print("📁 Files generated:")
    print("   - lmsys_analysis_results.json (detailed results)")
    print("   - lmsys_summary_report.txt (summary report)")
    print("   - *.png (visualization files)")

if __name__ == "__main__":
    main()
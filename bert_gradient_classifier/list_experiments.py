"""
Utility script to list all available experiments and groups.
"""

from experiment_config import (
    EXPERIMENT_CONFIGS,
    EXPERIMENT_GROUPS,
    list_all_experiments,
    list_all_groups
)


def print_experiment_details():
    """Print details of all experiments."""
    print("="*80)
    print("AVAILABLE EXPERIMENTS")
    print("="*80)
    print()
    
    for exp_name, config in EXPERIMENT_CONFIGS.items():
        print(f"Experiment: {exp_name}")
        print(f"  Description: {config.get('description', 'N/A')}")
        print(f"  Scene Model: {config.get('scene_model_name', 'N/A')}")
        print(f"  Scene Finetune: {config.get('scene_encoder_finetune', False)}")
        print(f"  Use Linguistic: {config.get('use_linguistic', False)}")
        if config.get('use_linguistic'):
            print(f"  Fusion Method: {config.get('fusion_method', 'N/A')}")
        print(f"  Sequence Model: {config.get('sequence_model_type', 'N/A')}")
        print(f"  Positional Encoding: {config.get('use_positional_encoding', False)}")
        print(f"  Classifier: {config.get('classifier_type', 'N/A')}")
        print()
    
    print(f"\nTotal experiments: {len(EXPERIMENT_CONFIGS)}")


def print_experiment_groups():
    """Print all experiment groups."""
    print("="*80)
    print("EXPERIMENT GROUPS")
    print("="*80)
    print()
    
    for group_name, experiments in EXPERIMENT_GROUPS.items():
        print(f"Group: {group_name}")
        print(f"  Experiments ({len(experiments)}):")
        for exp in experiments:
            print(f"    - {exp}")
        print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="List available experiments")
    parser.add_argument("--details", action="store_true", help="Show detailed experiment info")
    parser.add_argument("--groups", action="store_true", help="Show experiment groups")
    
    args = parser.parse_args()
    
    if args.details:
        print_experiment_details()
    elif args.groups:
        print_experiment_groups()
    else:
        # Show both
        print_experiment_groups()
        print()
        print_experiment_details()


if __name__ == "__main__":
    main()


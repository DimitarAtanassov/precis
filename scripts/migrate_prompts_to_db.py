#!/usr/bin/env python3
"""
Migrate prompts from prompts.yaml to floating_prompts database.

Usage:
    cd precis
    uv run python scripts/migrate_prompts_to_db.py

This will:
1. Read all prompts from prompts.yaml
2. Insert them into the floating_prompts database with version=1
3. Skip any prompts that already exist
"""

from pathlib import Path

import yaml
from floating_prompts import PromptRepository, get_session, init_db


def load_yaml_prompts(yaml_path: Path) -> dict:
    """Load prompts from YAML file."""
    with open(yaml_path) as f:
        return yaml.safe_load(f) or {}


def parse_prompt_entry(name: str, entry: dict | str) -> tuple[str | None, str]:
    """
    Parse a YAML prompt entry into (system_prompt, user_prompt).
    
    Handles both formats:
    - Full: { system_prompt: { prompt: "..." }, user_prompt: { prompt: "..." } }
    - Simple: { prompt: "..." } or just a string
    """
    if isinstance(entry, str):
        return None, entry
    
    if isinstance(entry, dict):
        # Full format with both prompts
        if "system_prompt" in entry and "user_prompt" in entry:
            system = entry["system_prompt"].get("prompt", "") if isinstance(entry["system_prompt"], dict) else ""
            user = entry["user_prompt"].get("prompt", "") if isinstance(entry["user_prompt"], dict) else ""
            return system or None, user
        
        # Simple format with just "prompt"
        if "prompt" in entry:
            return None, entry["prompt"]
    
    raise ValueError(f"Prompt '{name}' has unrecognized structure: {type(entry)}")


def migrate_prompts(yaml_path: Path, version: int = 1, dry_run: bool = False) -> None:
    """
    Migrate prompts from YAML to database.
    
    Args:
        yaml_path: Path to prompts.yaml
        version: Version number to use for all prompts (default: 1)
        dry_run: If True, just print what would be done without inserting
    """
    # Initialize database
    if not dry_run:
        init_db()
    
    # Load YAML prompts
    prompts = load_yaml_prompts(yaml_path)
    print(f"Found {len(prompts)} prompts in {yaml_path}")
    print("-" * 60)
    
    created = 0
    skipped = 0
    errors = 0
    
    with get_session() as session:
        repo = PromptRepository(session)
        
        for name, entry in prompts.items():
            try:
                system_prompt, user_prompt = parse_prompt_entry(name, entry)
                
                # Check if already exists
                if repo.exists(name, version=version):
                    print(f"⏭️  SKIP: '{name}' v{version} already exists")
                    skipped += 1
                    continue
                
                if dry_run:
                    print(f"🔍 DRY RUN: Would create '{name}' v{version}")
                    print(f"   System: {(system_prompt or '')[:50]}...")
                    print(f"   User: {user_prompt[:50]}...")
                    created += 1
                else:
                    repo.create(
                        name=name,
                        user_prompt=user_prompt,
                        system_prompt=system_prompt,
                        version=version,
                    )
                    print(f"✅ Created: '{name}' v{version}")
                    created += 1
                    
            except Exception as e:
                print(f"❌ ERROR: '{name}' - {e}")
                errors += 1
        
        if not dry_run:
            session.commit()
    
    print("-" * 60)
    print(f"Summary: {created} created, {skipped} skipped, {errors} errors")


def main() -> None:
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate prompts from YAML to database")
    parser.add_argument(
        "--yaml-path",
        type=Path,
        default=Path(__file__).parent.parent / "src/precis/prompts.yaml",
        help="Path to prompts.yaml",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Version number to use (default: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just print what would be done without inserting",
    )
    
    args = parser.parse_args()
    
    if not args.yaml_path.exists():
        print(f"Error: YAML file not found: {args.yaml_path}")
        return
    
    print(f"Migrating prompts to database (version={args.version})")
    if args.dry_run:
        print("🔍 DRY RUN MODE - no changes will be made")
    print()
    
    migrate_prompts(args.yaml_path, version=args.version, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

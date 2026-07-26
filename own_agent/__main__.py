"""python -m own_agent entry point."""

import asyncio
import sys

from own_agent.cli import run_cli


def main() -> int:
    args = [a for a in sys.argv[1:] if a]
    model: str | None = None
    prompt: str | None = None
    i = 0
    while i < len(args):
        if args[i] in ("-m", "--model") and i + 1 < len(args):
            model = args[i + 1]
            i += 2
        elif args[i] in ("-p", "--prompt") and i + 1 < len(args):
            prompt = args[i + 1]
            i += 2
        elif args[i] in ("-h", "--help"):
            print("Usage: python -m own_agent [--model NAME] [--prompt TEXT]")
            print("  --model, -m     Model name (e.g. deepseek/deepseek-chat)")
            print("  --prompt, -p    Run a single prompt non-interactively")
            return 0
        else:
            i += 1

    asyncio.run(run_cli(model=model, prompt=prompt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command line: ``tunai-agent chat``, ``tunai-agent ask``, ``tunai-agent models``."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import openai

from .agent import AgentResult
from .client import TunAIClient
from .config import ConfigError, Settings
from .errors import TunAIError, describe_error
from .support_agent import DEFAULT_KB_DIR, DEFAULT_ORDERS, build_support_agent

DIM, RESET = "\033[2m", "\033[0m"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format=f"{DIM}%(message)s{RESET}",
    )
    # One line per HTTP request would bury the tool trace that -v is for.
    for noisy in ("httpx", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        settings = Settings.from_env()
        with TunAIClient(settings) as client:
            return args.command(client, args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
    except (openai.APIError, TunAIError) as exc:
        print(describe_error(exc), file=sys.stderr)
    except KeyboardInterrupt:
        print()
        return 130
    return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tunai-agent", description="A customer-support agent built on the TunAI API."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="show tool calls and usage")
    commands = parser.add_subparsers(required=True, metavar="command")

    def agent_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--model", help="chat model id (default: TUNAI_CHAT_MODEL)")
        command.add_argument("--max-steps", type=int, default=6, help="model requests per turn")
        command.add_argument("--kb-dir", default=DEFAULT_KB_DIR, help="folder of .md documents")
        command.add_argument("--orders", default=DEFAULT_ORDERS, help="orders JSON file")

    chat = commands.add_parser("chat", help="talk to the agent interactively")
    agent_options(chat)
    chat.set_defaults(command=_chat)

    ask = commands.add_parser("ask", help="ask one question and exit")
    agent_options(ask)
    source = ask.add_mutually_exclusive_group(required=True)
    source.add_argument("question", nargs="?", help="the question")
    source.add_argument("--audio", help="an audio file to transcribe and ask")
    ask.add_argument("--language", help="audio language hint, e.g. ar, fr, en")
    ask.set_defaults(command=_ask)

    models = commands.add_parser("models", help="list the models your key can use")
    models.set_defaults(command=_models)
    return parser


def _build(client: TunAIClient, args: argparse.Namespace):
    return build_support_agent(
        client,
        kb_dir=args.kb_dir,
        orders_file=args.orders,
        model=args.model,
        max_steps=args.max_steps,
        on_event=_trace if args.verbose else None,
    )


def _ask(client: TunAIClient, args: argparse.Namespace) -> int:
    question = args.question
    if args.audio:
        question = client.transcribe(args.audio, language=args.language)
        print(f"{DIM}transcript: {question}{RESET}")
    result = _build(client, args).run(question)
    _print_answer(result, args.verbose)
    return 0


def _chat(client: TunAIClient, args: argparse.Namespace) -> int:
    agent = _build(client, args)
    history: list[dict[str, Any]] = []
    print(f"Dar Jasmin support · {agent.model} · Ctrl+D to quit\n")
    while True:
        try:
            question = input("you › ").strip()
        except EOFError:
            print()
            return 0
        if not question:
            continue
        try:
            result = agent.run(question, history)
        except (openai.APIError, TunAIError) as exc:
            # One failed turn should not end the conversation.
            print(describe_error(exc), file=sys.stderr)
            continue
        _print_answer(result, args.verbose, prefix="agent › ")
        history += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": result.answer},
        ]


def _models(client: TunAIClient, args: argparse.Namespace) -> int:
    for model_id in client.list_models():
        print(model_id)
    return 0


def _print_answer(result: AgentResult, verbose: bool, prefix: str = "") -> None:
    print(f"{prefix}{result.answer}\n")
    if verbose:
        usage = result.usage
        note = " · stopped at max steps" if result.stopped_early else ""
        print(
            f"{DIM}{usage.requests} requests · {len(result.tool_calls)} tool calls · "
            f"{usage.prompt_tokens:,} in / {usage.completion_tokens:,} out tokens{note}{RESET}\n"
        )


def _trace(event: str, data: dict[str, Any]) -> None:
    if event == "tool_call":
        print(f"{DIM}  → {data['name']}({data['arguments']}){RESET}")
    elif event == "tool_result":
        preview = data["result"] if len(data["result"]) < 160 else data["result"][:157] + "..."
        print(f"{DIM}  ← {preview} ({data['duration_ms']} ms){RESET}")


if __name__ == "__main__":
    raise SystemExit(main())

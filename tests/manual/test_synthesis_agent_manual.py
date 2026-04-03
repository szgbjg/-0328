import asyncio

from tests.manual.manual_runner import parse_args, run_manual_test


def main():
    args = parse_args("synthesis_agent")
    code = asyncio.run(
        run_manual_test(
            role="synthesis_agent",
            model=args.model,
            save=args.save,
            verbose=args.verbose,
        )
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()

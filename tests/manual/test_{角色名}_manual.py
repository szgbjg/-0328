import asyncio

from tests.manual.manual_runner import parse_args, run_manual_test


def main():
    # 占位脚本默认跑 question_creator，建议优先使用具体角色脚本。
    args = parse_args("question_creator")
    code = asyncio.run(
        run_manual_test(
            role="question_creator",
            model=args.model,
            save=args.save,
            verbose=args.verbose,
        )
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()

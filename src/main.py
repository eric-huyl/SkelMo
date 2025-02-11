# my_cli_tool/main.py
import argparse


def main():
    # 创建命令行解析器
    parser = argparse.ArgumentParser(description="My CLI Tool")

    # 添加命令行参数
    parser.add_argument('-n',
                        '--name',
                        type=str,
                        help="Your name",
                        required=True)
    parser.add_argument('-g',
                        '--greet',
                        action='store_true',
                        help="Greet the user")

    # 解析命令行参数
    args = parser.parse_args()

    # 执行命令
    if args.greet:
        print(f"Hello, {args.name}!")
    else:
        print(f"Goodbye, {args.name}!")


if __name__ == "__main__":
    main()

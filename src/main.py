# my_cli_tool/main.py
import argparse
from video import video_recognition
from image import image_recognition
from stream import stream_recognition


def main():
    parser = argparse.ArgumentParser(
        description="Skeleton Motion Recognition Tool")
    parser.add_argument('input_file', type=str,
                        default='input.mp4', help='Target file path')
    parser.add_argument('-m',
                        '--mode',
                        type=str,
                        help='Set working mode (video/image/stream)')
    parser.add_argument('-fi', '--frame_interval', type=int,
                        default=10,
                        help='Frame interval for video recognition')
    parser.add_argument('-o', '--output_file', type=str,
                        default='output.json',
                        help='Output file path (name.json)')
    args = parser.parse_args()

    if not args.input_file:
        print("Please specify the input file path.")
        return

    mode_functions = {
        'video': lambda: video_recognition(args.input_file,
                                           args.frame_interval,
                                           args.output_file),
        'image': lambda: image_recognition(args.input_file,
                                           args.output_file),
        'stream': lambda: stream_recognition(args.input_file)
    }

    try:
        if args.mode in mode_functions:
            mode_functions[args.mode]()
        else:
            print("Please specify the working mode (video/image/stream).")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

import argparse
import charex

parser = argparse.ArgumentParser(
    description="Character Extraction from Handwriting Tool",
)

parser.add_argument(
    "-p",
    "--preview",
    action="store_true",
    help="show preview even if [output] is set",
)

parser.add_argument(
    "-f",
    "--fragmentation",
    action="store_true",
    help="try to split touching letters",
)

parser.add_argument(
    "input",
    type=str,
    help="path to the input image file",
)

parser.add_argument(
    "output",
    nargs="?",
    type=str,
    help="path to the output directory",
)


def main():
    args = parser.parse_args()
    input_file = args.input
    output_dir = args.output
    preview_flag = args.preview
    fragmentation_flag = args.fragmentation
    charex.processor.process(
        input_file=input_file,
        output_dir=output_dir,
        preview_flag=preview_flag,
        fragmentation_flag=fragmentation_flag,
    )


if __name__ == "__main__":
    main()

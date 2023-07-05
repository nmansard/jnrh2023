#!/usr/bin/env python3
"""Cut python files in bits loadable by ipython."""

from pathlib import Path
import json

hashtags = ["jupyter_snippet"]


def generate_from_id(tp_id: int):
    folder = Path() / f"tp{tp_id}"
    ipynb = next(Path().glob(f"{tp_id}_*.ipynb"))
    generate(ipynb, folder)


def generate(ipynb, folder):
    print(f"processing {ipynb} with scripts in {folder}")
    with ipynb.open() as f:
        data = json.load(f)
    cells_copy = data["cells"].copy()
    generated = folder / "generated"
    generated.mkdir(exist_ok=True)
    for filename in folder.glob("*.py"):
        print(f" processing {filename}")
        content = []
        hidden = False
        dest = None
        with filename.open() as f_in:
            for line_number, line in enumerate(f_in):
                if any([f"# %{hashtag}" in line for hashtag in hashtags]):
                    if dest is not None:
                        raise SyntaxError(
                            f"%{hashtags[0]} block open twice at line {line_number + 1}"
                        )
                    dest = generated / f"{filename.stem}_{line.split()[2]}"
                    hidden = False
                elif any([line.strip() == f"# %end_{hashtag}" for hashtag in hashtags]):
                    if dest is None:
                        raise SyntaxError(
                            f"%{hashtags[0]} block before open at line {line_number + 1}"
                        )
                    with dest.open("w") as f_out:
                        f_out.write("".join(content))
                    for cell_number, cell in enumerate(cells_copy):
                        if len(cell["source"]) == 0:
                            continue
                        if cell["source"][0].endswith(f"%load {dest}"):
                            data["cells"][cell_number]["source"] = [
                                f"# %load {dest}\n"
                            ] + content
                        # if f'%do_not_load {dest}' in cell['source'][0]:
                        #    data['cells'][cell_number]['source'] = [f'%do_not_load {dest}\n']
                    content = []
                    hidden = False
                    dest = None
                elif dest is not None:
                    content.append(line)
    with ipynb.open("w") as f:
        f.write(json.dumps(data, indent=1))


if __name__ == "__main__":
    for tp_number in [0, 2, 3, 4, 5]:
        generate_from_id(tp_number)

    for app in ["appendix_scipy_optimizers"]:
        generate(next(Path().glob(app + ".ipynb")), Path() / "appendix")

import pathlib

import typer
import webbrowser
import os

from highlighting.highlighter import BertHighlighter
from highlighting import DATA_FOLDER

app = typer.Typer()


@app.command()
def get_data(data_folder: pathlib.Path = typer.Option(DATA_FOLDER, file_okay=False)):
    ...


@app.command()
def train(data_folder: pathlib.Path = typer.Option(DATA_FOLDER, file_okay=False, exists=True)):
    ...


@app.command()
def highlight(question: str, answer: str):
    highlighter = BertHighlighter.from_pretrained()

    html = highlighter.highlight(question=question, answer=answer)

    filename = f'{hash(answer)}.html'

    with open(filename, 'w') as f:
        html = f"""<h3>{question}</h3>
<p>{html}</p>
"""
        f.write(html)

    webbrowser.open_new_tab(f'file:///{os.getcwd()}/{filename}')


if __name__ == "__main__":
    app()

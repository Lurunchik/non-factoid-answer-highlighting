import logging
import pathlib
import tempfile
import webbrowser
from typing import List, Optional

import typer

from highlighting.data import DATA_FOLDER
from highlighting.dataset import download_nfl6_dataset, prepare_nfl6_dataset
from highlighting.highlighter import BertHighlighter
from highlighting.train import train_model

LOGGER = logging.getLogger('highlighting')

app = typer.Typer()


@app.command()
def load_data(
    data_folder: pathlib.Path = typer.Option(
        DATA_FOLDER, file_okay=False, help='Path to the folder where the dataset will be saved'
    )
):
    download_nfl6_dataset(data_folder)
    prepare_nfl6_dataset(data_folder)


@app.command()
def train(
    data_folder: pathlib.Path = typer.Option(
        DATA_FOLDER, file_okay=False, exists=True, help='Path to the folder where the dataset is stored'
    ),
    gpus: List[int] = typer.Option(None, help='List of GPUs to use, skip to train on CPU'),
):
    train_model(data_folder=data_folder, gpus=list(gpus) or None)


@app.command()
def highlight(
    question: str = typer.Option(..., '-q', '--question', help='Question to highlight attention from'),
    answer: str = typer.Option(..., '-a', '--answer', help='Answer to highlight'),
):
    highlighter = BertHighlighter.from_pretrained()

    html_highlighting = highlighter.highlight(question=question, answer=answer)
    html_highlighting = f'<h3>{question}</h3><p>{html_highlighting}</p>'

    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as file:
        file.write(html_highlighting)

    webbrowser.open_new_tab(f'file:///{file.name}')


if __name__ == '__main__':
    app()

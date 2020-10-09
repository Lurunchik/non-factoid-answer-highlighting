import os
import pathlib
import webbrowser

import typer

from highlighting import DATA_FOLDER
from highlighting.dataset import prepare_dataset
from highlighting.highlighter import BertHighlighter

app = typer.Typer()


@app.command()
def load_dataset(data_folder: pathlib.Path = typer.Option(DATA_FOLDER, file_okay=False)):
    prepare_dataset(data_folder)


@app.command()
def train(data_folder: pathlib.Path = typer.Option(DATA_FOLDER, file_okay=False, exists=True)):
    import highlighting.train

    highlighting.train.train(
        train_path=data_folder / 'train.joblib',
        val_path=data_folder / 'val.joblib',
        test_path=data_folder / 'test.joblib',
    )


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


if __name__ == '__main__':
    app()

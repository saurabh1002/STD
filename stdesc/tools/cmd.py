# MIT License
#
# Copyright (c) 2023 Saurabh Gupta, Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch,
# Cyrill Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from pathlib import Path
from typing import Optional

import typer

from stdesc.datasets import available_dataloaders


def name_callback(value: str):
    if not value:
        return value
    dl = available_dataloaders()
    if value not in dl:
        raise typer.BadParameter(f"Supported dataloaders are:\n{', '.join(dl)}")
    return value


app = typer.Typer(add_completion=False, rich_markup_mode="rich")

# Remove from the help those dataloaders we explicitly say how to use
_available_dl_help = available_dataloaders()

docstring = f"""
:STDesc:\n
\b
[bold green]Examples: [/bold green]
# Use a specific dataloader: {", ".join(_available_dl_help)}
$ stdesc_pipeline --dataloader mulran --gt-overlap-threshold 0.5 --visualize <path-to-mulran-root> <path-to-save-results>:
"""


@app.command(help=docstring)
def stdesc_pipeline(
    data: Path = typer.Argument(
        ...,
        help="The data directory used by the specified dataloader",
        show_default=False,
    ),
    results_dir: Path = typer.Argument(
        ...,
        help="The path where results are to be stored",
        show_default=False,
        exists=False,
    ),
    dataloader: str = typer.Option(
        None,
        show_default=False,
        case_sensitive=False,
        autocompletion=available_dataloaders,
        callback=name_callback,
        help="[Optional] Use a specific dataloader from those supported",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        exists=True,
        show_default=False,
        help="[Optional] Path to the configuration file",
    ),
    # Aditional Options ---------------------------------------------------------------------------
    sequence: Optional[str] = typer.Option(
        None,
        "--sequence",
        "-s",
        show_default=False,
        help="[Optional] For some dataloaders, you need to specify a given sequence",
        rich_help_panel="Additional Options",
    ),
):
    # Lazy-loading for faster CLI
    from stdesc.datasets import dataset_factory
    from stdesc.pipeline import STDescPipeline

    STDescPipeline(
        dataset=dataset_factory(
            dataloader=dataloader,
            data_dir=data,
            # Additional options
            sequence=sequence,
        ),
        results_dir=results_dir,
        config=config,
    ).run().print()


def run():
    app()

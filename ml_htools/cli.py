# -*- coding: utf-8 -*-

"""Console script for ml_htools."""
import sys
import click


@click.command()
def main(args=None):
    """Console script for ml_htools."""
    click.echo("Replace this message by putting your code into "
               "ml_htools.cli.main")
    click.echo("See click documentation at http://click.pocoo.org/")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover

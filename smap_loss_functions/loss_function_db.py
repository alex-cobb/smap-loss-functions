"""Loss function SQLite database code"""

from .loss_function import LossFunction


def set_up_loss_function_db(out_connection):
    """Set up database for loss functions

    Creates table to store loss functions.

    """
    cursor = out_connection.cursor()
    cursor.execute("""
    CREATE TABLE loss_function (
      ease_col integer NOT NULL,
      ease_row integer NOT NULL,
      Wmin real NOT NULL,
      Wmax real NOT NULL,
      LA real NOT NULL,
      LB real NOT NULL,
      LC real NOT NULL,
      rmse real NULL,
      PRIMARY KEY (ease_col, ease_row)
    )""")
    cursor.close()


def get_loss_function_from_db(connection, ease_col, ease_row):
    """Instantiate loss function for an EASE column and row from database

    Returns loss function and Wmax, which is needed for simulations.

    The parameters are computed based on the procedure of Koster et al (2017) and are
    therefore ignored.

    """
    parameters = connection.execute(
        """
    SELECT Wmin, Wmax, LA, LB, LC
    FROM loss_function
    WHERE ease_col = ? AND ease_row = ?""",
        (ease_col, ease_row),
    ).fetchone()
    if parameters is None:
        min_col, max_col, min_row, max_row = connection.execute(
            'SELECT min(ease_col), max(ease_col), min(ease_row), max(ease_row) '
            'FROM loss_function'
        ).fetchone()
        raise ValueError(
            f'EASE col={ease_col} row={ease_row} not in ranges: '
            f'{min_col}--{max_col}, {min_row}--{max_row}'
        )
    Wmin, Wmax, LA, LB, LC = parameters
    return LossFunction(Wmax, Wmin, LA, LB, LC)

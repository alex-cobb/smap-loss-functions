"""Write SMAP and IMERG data to an SQLite database"""


def write_smap_db(smap_infile, imerg_infile, connection):
    """Write SMAP and IMERG data to an SQLite database"""
    cursor = connection.cursor()
    cursor.execute("""
    CREATE TABLE smap_data (
      start_datetime timestamp NOT NULL,
      thru_datetime timestamp NOT NULL,
      ease_col integer NOT NULL,
      ease_row integer NOT NULL,
      soil_moisture real NOT NULL,
      PRIMARY KEY (start_datetime, ease_col, ease_row)
    )""")
    smap_header = next(smap_infile).strip()
    expected_header = 'start_datetime,thru_datetime,column,row,value'
    assert smap_header == expected_header, (
        f'Bad SMAP data file header: expected {expected_header}, got {smap_header}'
    )
    del smap_header
    cursor.executemany(
        """
    INSERT INTO smap_data
      (start_datetime, thru_datetime, ease_col, ease_row, soil_moisture)
    VALUES (?, ?, ?, ?, ?)""",
        (row.split(',') for row in smap_infile),
    )

    cursor.execute("""
    CREATE TABLE imerg_data (
      start_datetime timestamp NOT NULL,
      thru_datetime timestamp NOT NULL,
      ease_col integer NOT NULL,
      ease_row integer NOT NULL,
      precipitation real NOT NULL,
      PRIMARY KEY (start_datetime, ease_col, ease_row)
    )""")
    imerg_header = next(imerg_infile).strip()
    assert imerg_header == expected_header, (
        f'Bad IMERG data file header: expected {expected_header}, got {imerg_header}'
    )
    cursor.executemany(
        """
    INSERT INTO imerg_data
      (start_datetime, thru_datetime, ease_col, ease_row, precipitation)
    VALUES (?, ?, ?, ?, ?)""",
        (row.split(',') for row in imerg_infile),
    )
    cursor.close()
    return 0

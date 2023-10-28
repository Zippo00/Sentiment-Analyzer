'''
Functions for SQLite Database management.
'''

import sqlite3

def sql_execute(query, db, *args):
    '''
    Execute a SQLite Database Query.
    Can be used to create a new table, etc.
    
    EXAMPLE QUERY FOR CREATING A TABLE:
    "CREATE TABLE example_name (
                    pricelevel INTEGER,
                    orders REAL,
                    timestamp INTEGER)"
    EXAMPLE QUERY FOR DELETING TABLE DATA:
    "DELETE FROM example_name"
    :param query: (str) Query to execute.
    :param db: (str) Database to use.
    '''
    try:
        # Establish connection to DB
        sqlite_connection = sqlite3.connect(f"data/{db}")
        print("Connected to SQLite Database.")
        # Create a cursor object
        sql_cursor = sqlite_connection.cursor()
        # Execute the query
        if args:
            sql_cursor.execute(query, args)
        else:
            sql_cursor.execute(query)
        sqlite_connection.commit()
        print("SQL Query Executed.")
    except sqlite3.Error as e:
        print("An error realted to SQLite DB occured - ", e)
        print(f"Query that caused error: {query}")
    finally:
        # Close the connection
        if sqlite_connection:
            sqlite_connection.close()
            print("Disconnected from SQLite Database.")

def fetch_data(table, db, query=None):
    '''
    Fetch a certain table from SQLite Database.
    :param table: (str) Name of the table to fetch.
    :param db: (str) Database to use e.g. 'mnemos.db'
    :(Optional) param query: (str) Optional query to execute. If None, fetches all data from the given table.
    :return: Database Table.
    '''
    try:
        # Establish connection to DB
        sqlite_connection = sqlite3.connect(f"data/{db}")
        print("Connected to SQLite Database.")
        # Save the preferred query
        if query:
            sql_query = query
        else:
            sql_query = f"SELECT * FROM {table}"
        # Create a cursor object
        sql_cursor = sqlite_connection.cursor()
        # Execute query
        sql_cursor.execute(sql_query)
        db_table = sql_cursor.fetchall()
        print("Fetched a Table from SQLite Database.")
    except sqlite3.Error as e:
        print("An error realted to SQLite DB occured - ", e)
    finally:
        # Close the connection
        if sqlite_connection:
            sqlite_connection.close()
            print("Disconnected from SQLite Database.")
    return db_table

def fetch_all_tables(db):
    '''
    Fetch the names of all tables in given database and print them.
    :param db: (str) Database to use e.g. 'mnemos.db'
    '''
    try:
        # Establish connection to DB
        sqlite_connection = sqlite3.connect(f"data/{db}")
        print("Connected to SQLite Database.")
        # Save the preferred query
        sql_query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        # Create a cursor object
        sql_cursor = sqlite_connection.cursor()
        # Execute query
        sql_cursor.execute(sql_query)
        print(f"List of all tables in database:\n{sql_cursor.fetchall()}")
    except sqlite3.Error as e:
        print("An error realted to SQLite DB occured - ", e)
    finally:
        # Close the connection
        if sqlite_connection:
            sqlite_connection.close()
            print("Disconnected from SQLite Database.")

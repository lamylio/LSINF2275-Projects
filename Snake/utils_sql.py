def create_table_if_not_exists(cursor, table):
    from pandas import DataFrame
    try:
        DataFrame(columns=[0,1,2,3,"alpha"]).to_sql(table, cursor.connection, if_exists="fail", dtype="float")
        print("TABLE", table, "WAS CREATED!")
    except ValueError:
        print("TABLE", table, "ALREADY EXISTS!")
    except:
        raise Warning("An error occured while performing create_table")
    

def insert_new_state(cursor, table, state):
    cursor.execute('''INSERT INTO "{}" VALUES (?, 0, 0, 0, 0, 1)'''.format(table), [state])

def get_values_from_state(cursor, table, state, default=(0,0,0,0,1)):
    try:
        cursor.execute('''SELECT * FROM "{}" WHERE "index" = ?'''.format(table), [state])
        results = cursor.fetchone()
        if results is not None: return results[1:]
        else: return default
    except:
        raise Warning("An error occured while performing get_values_from_state")

def get_table_length(cursor, table):
    cursor.execute('''SELECT COUNT(*) FROM "{}"'''.format(table))
    return cursor.fetchone()[0]

def update_value_from_state(cursor, table, state, column, value):
    cursor.execute('''UPDATE "{}" SET "{}"={} WHERE "index"=?'''.format(table, str(column), value), [str(state)])

def save_changes(database):
    database.commit()
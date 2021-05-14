from numpy import append

def insert_new_state(cursor, table, state):
    cursor.execute('''INSERT INTO "{}" VALUES (?, 0, 0, 0, 0)'''.format(table), [state])

def get_values_from_state(cursor, table, state, default=(0,0,0,0)):
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

def update_value_from_state_and_action(cursor, table, state, action, value):
    cursor.execute('''UPDATE "{}" SET "{}"={} WHERE "index"=?'''.format(table, str(action), value), [str(state)])

def save_changes(database):
    database.commit()
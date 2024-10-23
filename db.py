import pymysql

# Function to establish a connection with the MySQL database
def get_db_connection():
    return pymysql.connect(
        host="localhost",
        user="****",
        password="***",  # Replace with your MySQL password
        database="*****"  # Replace with your database name
    )

# Function to add a new user to the database
def add_new_user(age, gender, height, weight, active_hours, workout_type, workout_hours, plan_type, kg_change, adjusted_tdee):
    conn = get_db_connection()
    cursor = conn.cursor()
    sql = '''
    INSERT INTO users (age, gender, height, weight, active_hours, workout_type, workout_hours, plan_type, kg_change, adjusted_tdee, current_day)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 1)
    '''
    cursor.execute(sql, (age, gender, height, weight, active_hours, workout_type, workout_hours, plan_type, kg_change, adjusted_tdee))
    conn.commit()
    user_id = cursor.lastrowid  # Get the new user's ID
    conn.close()
    return user_id
def get_user_food_logs(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    sql = '''
    SELECT day, SUM(calories) AS total_calories
    FROM food_logs
    WHERE user_id = %s
    GROUP BY day
    ORDER BY day
    '''
    cursor.execute(sql, (user_id,))
    food_logs = cursor.fetchall()
    conn.close()
    return food_logs

# Function to retrieve user data based on user ID
def get_user_data(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    sql = 'SELECT * FROM users WHERE user_id = %s'
    cursor.execute(sql, (user_id,))
    user_data = cursor.fetchone()  # Fetch the user's data
    conn.close()
    return user_data

# Function to log food for a user
def add_food_log(user_id, day, food_name, weight, calories, protein, carbs, fat):
    conn = get_db_connection()
    cursor = conn.cursor()
    sql = '''
    INSERT INTO food_logs (user_id, day, food_name, weight, calories, protein, carbs, fat)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    '''
    cursor.execute(sql, (user_id, day, food_name, weight, calories, protein, carbs, fat))
    conn.commit()
    conn.close()

# Function to update the current day for the user
def update_user_day(user_id, current_day):
    conn = get_db_connection()
    cursor = conn.cursor()
    sql = 'UPDATE users SET current_day = %s WHERE user_id = %s'
    cursor.execute(sql, (current_day, user_id))
    conn.commit()
    conn.close()

# Get the last recorded day for a user
def get_last_logged_day(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    sql = '''
    SELECT MAX(day) FROM food_logs WHERE user_id = %s
    '''
    cursor.execute(sql, (user_id,))
    last_day = cursor.fetchone()[0]  
    conn.close()
    return last_day if last_day else 0 

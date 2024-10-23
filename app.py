import streamlit as st
import numpy as np
import pandas as pd
from db import get_db_connection, add_food_log, update_user_day, get_user_data, add_new_user,get_user_food_logs
from food_recognition import predict_food
from volume_estimation import estimate_food_volume
from PIL import Image
import pickle
from db import get_last_logged_day 
import matplotlib.pyplot as plt

# Load datasets
indian_food_nutrition = pd.read_csv('Dataset/Indian Food Nutrition Dataset.csv')

# Load models for TDEE prediction
@st.cache_resource 
def load_models():
    with open('BMR_model.pkl', 'rb') as f:
        bmr_model = pickle.load(f)
    
    with open('noworkout.pkl', 'rb') as f:
        noworkout_model = pickle.load(f)
    
    with open('workout.pkl', 'rb') as f:
        workout_model = pickle.load(f)
    
    return bmr_model, noworkout_model, workout_model

bmr_model, noworkout_model, workout_model = load_models()

def visualize_user_progress(user_id):
    # Fetch user food logs
    food_logs = get_user_food_logs(user_id)

    if not food_logs:
        st.write("No food logs available to display.")
        return

    # Extract days and calories for plotting
    days = [log[0] for log in food_logs]  # This should be the actual day numbers
    daily_calories = [log[1] for log in food_logs]

    # Plot user calorie consumption over time
    fig, ax = plt.subplots(figsize=(6, 4))
    threshold = st.session_state.adjusted_tdee
    plan_type = st.session_state.plan_type

    colors = []
    for calories in daily_calories:
        if plan_type == 'Weight Loss':
            if calories < threshold:
                colors.append('green')
            elif calories > threshold:
                colors.append('red')
            else:
                colors.append('yellow')
        else:  # Weight Gain
            if calories > threshold:
                colors.append('green')
            elif calories < threshold:
                colors.append('red')
            else:
                colors.append('yellow')

    # Scatter plot with the correct x-axis (actual days)
    ax.scatter(days, daily_calories, c=colors, s=50)  
    ax.plot(days, daily_calories, label="Daily Calories", color="blue")
    ax.axhline(threshold, color='gray', linestyle='--', label=f'Threshold: {threshold} kcal')

    # Set x-axis labels to be actual days
    ax.set_xticks(days)  # Make sure x-axis ticks correspond to actual days
    ax.set_xlabel('Day')
    ax.set_ylabel('Calories')
    ax.set_title(f'Calorie Consumption over Time (User {user_id})')
    ax.legend()

    # Display the plot
    st.pyplot(fig)

# Function to calculate weight change based on caloric surplus/deficit
def calculate_weight_change(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    sql = '''
    SELECT SUM(calories)
    FROM food_logs
    WHERE user_id = %s
    '''
    cursor.execute(sql, (user_id,))
    total_calories = cursor.fetchone()[0]
    conn.close()

    if total_calories:
        # Calculate weight change
        tdee = st.session_state.adjusted_tdee
        calorie_difference = total_calories - (tdee * len(get_user_food_logs(user_id)))
        kg_change = calorie_difference / 7700  # 7700 kcal = 1 kg

        if st.session_state.plan_type == 'Weight Loss':
            if kg_change > 0:
                st.write(f"You have exceeded your goal by consuming {kg_change:.2f} kg worth of extra calories.")
            else:
                st.write(f"Great job! You have reduced {abs(kg_change):.2f} kg based on your calorie deficit.")
        else:
            if kg_change > 0:
                st.write(f"Great job! You have gained {kg_change:.2f} kg based on your calorie surplus.")
            else:
                st.write(f"You have underconsumed by {abs(kg_change):.2f} kg worth of calories for weight gain.")


def tracker_section():
    # Display Tracker Section after user TDEE is calculated or for existing users
    if 'user_id' in st.session_state:
        st.write(f"Welcome to Day {st.session_state.current_day}, User ID: {st.session_state.user_id}")
        st.write(f"Your adjusted TDEE is {st.session_state.adjusted_tdee} kcal per day")
        st.write(f"Plan Type: {st.session_state.plan_type}")

        # --- TRACKER SECTION ---
        st.title('Your Nutrition Plan')
        st.subheader(f"Adjusted TDEE: {st.session_state.adjusted_tdee} kcal/day")
        st.subheader(f"Plan: {st.session_state.plan_type}")
        st.write('Welcome to your personalized nutrition plan.')

        # Initialize session state for tracking foods and days
        if 'foods' not in st.session_state:
            st.session_state.foods = []
        if 'cumulative_nutrition' not in st.session_state:
            st.session_state.cumulative_nutrition = {}

        # Function to calculate cumulative nutrition
        def calculate_cumulative_nutrition():
            cumulative_calories = 0
            cumulative_protein = 0
            cumulative_carbs = 0
            cumulative_fat = 0
            
            for food in st.session_state.foods:
                if food['day'] == st.session_state.current_day:
                    cumulative_calories += food['calories']
                    cumulative_protein += food['protein']
                    cumulative_carbs += food['carbs']
                    cumulative_fat += food['fat']

            return cumulative_calories, cumulative_protein, cumulative_carbs, cumulative_fat

        # Display current day
        st.write(f'---\n### Day {st.session_state.current_day}')

        # Add food by image for the current day
        if st.button(f'+ Add Food for Day {st.session_state.current_day}', key=f'add_food_{st.session_state.current_day}_{len(st.session_state.foods) + 1}'):
            st.session_state.show_file_uploader = True

        # If the "Add Food by Image" button has been pressed, show the file uploader
        if 'show_file_uploader' in st.session_state and st.session_state.show_file_uploader:
            st.write(f"Upload your food image for Day {st.session_state.current_day} (only .jpg images are allowed):")
            food_image = st.file_uploader(f"Choose a .jpg image for Day {st.session_state.current_day}...", type=["jpg"], key=f'file_uploader_{st.session_state.current_day}')
            
            if food_image is not None:
                # Display the image
                st.image(food_image, caption=f"Uploaded image for Day {st.session_state.current_day}", use_column_width=True)
                
                # Predict food
                predicted_food = predict_food(food_image)
                st.write(f'Predicted Food for Day {st.session_state.current_day}: {predicted_food}')

                # Use the predicted food to check if it exists in the dataset
                if predicted_food in indian_food_nutrition['Food Name'].values:
                    st.write(f'**Food found in the database for Day {st.session_state.current_day}:** {predicted_food}')
                    
                    # Fetch the nutritional values for the predicted food
                    nutrition_info = indian_food_nutrition[indian_food_nutrition['Food Name'] == predicted_food].iloc[0]
                    
                    # Estimate food volume
                    image = Image.open(food_image)
                    open_cv_image = np.array(image)  
                    open_cv_image = open_cv_image[:, :, ::-1].copy() 

                    volume = estimate_food_volume(open_cv_image)
                    st.write(f'**Estimated Volume for Day {st.session_state.current_day}:** {volume:.2f} cubic centimeters')

                    # Convert volume to grams
                    conversion_factor = nutrition_info.get('Volume to Weight Conversion (g/cm³)', 1.0)  
                    weight_in_grams = volume * conversion_factor
                    st.write(f'**Estimated Weight for Day {st.session_state.current_day}:** {weight_in_grams:.2f} grams')

                    # Calculate nutrition details
                    calories = (nutrition_info['Calories (per 100g)'] * weight_in_grams) / 100
                    protein = (nutrition_info['Protein (g per 100g)'] * weight_in_grams) / 100
                    carbs = (nutrition_info['Carbs (g per 100g)'] * weight_in_grams) / 100
                    fat = (nutrition_info['Fats (g per 100g)'] * weight_in_grams) / 100

                    # Store the food and its nutrition in session state
                    st.session_state.foods.append({
                        'day': st.session_state.current_day,
                        'name': predicted_food,
                        'weight': weight_in_grams,
                        'calories': calories,
                        'protein': protein,
                        'carbs': carbs,
                        'fat': fat
                    })

                    add_food_log(st.session_state.user_id, st.session_state.current_day, predicted_food, weight_in_grams, calories, protein, carbs, fat)

                    st.success(f"Food log for {predicted_food} has been added to the database.")

                    # Reset file uploader for the next food
                    st.session_state.show_file_uploader = False

                else:
                    st.write(f'**Food not found in the database for Day {st.session_state.current_day}:** {predicted_food}')

        # Compute Nutrition button
        if st.button(f'Compute Nutrition'):
            # Calculate and display cumulative nutrition for the current day
            cumulative_calories, cumulative_protein, cumulative_carbs, cumulative_fat = calculate_cumulative_nutrition()

            st.write(f"### Cumulative Nutrition for Day {st.session_state.current_day}:")
            st.write(f"- **Total Calories:** {cumulative_calories:.2f} kcal")
            st.write(f"- **Total Protein:** {cumulative_protein:.2f} g")
            st.write(f"- **Total Carbs:** {cumulative_carbs:.2f} g")
            st.write(f"- **Total Fat:** {cumulative_fat:.2f} g")

        # End Day's Consumption button (dynamic day)
        if st.button(f'End Day {st.session_state.current_day} Consumption'):
            # Clear food data for the current day and increment the day
            cumulative_calories, cumulative_protein, cumulative_carbs, cumulative_fat = calculate_cumulative_nutrition()

            # Store the cumulative nutrition for the current day
            st.session_state.cumulative_nutrition[st.session_state.current_day] = {
                'calories': cumulative_calories,
                'protein': cumulative_protein,
                'carbs': cumulative_carbs,
                'fat': cumulative_fat
            }

            st.write(f"### Cumulative Nutrition for Day {st.session_state.current_day}:")
            st.write(f"- **Total Calories:** {cumulative_calories:.2f} kcal")
            st.write(f"- **Total Protein:** {cumulative_protein:.2f} g")
            st.write(f"- **Total Carbs:** {cumulative_carbs:.2f} g")
            st.write(f"- **Total Fat:** {cumulative_fat:.2f} g")

            # Feedback based on the user's plan and TDEE
            if st.session_state.plan_type == 'Weight Loss':
                if cumulative_calories < st.session_state.adjusted_tdee:
                    st.success(f"Great job! You've consumed {cumulative_calories:.2f} kcal, which is below your TDEE of {st.session_state.adjusted_tdee} kcal for weight loss.")
                else:
                    st.warning(f"You've consumed {cumulative_calories:.2f} kcal, which is above your TDEE of {st.session_state.adjusted_tdee} kcal. Try to adjust your intake for weight loss.")
            elif st.session_state.plan_type == 'Weight Gain':
                if cumulative_calories > st.session_state.adjusted_tdee:
                    st.success(f"Great job! You've consumed {cumulative_calories:.2f} kcal, which is above your TDEE of {st.session_state.adjusted_tdee} kcal for weight gain.")
                else:
                    st.warning(f"You've consumed {cumulative_calories:.2f} kcal, which is below your TDEE of {st.session_state.adjusted_tdee} kcal. You may need to increase your intake for weight gain.")

            # Increment day and clear the current day's food data
            st.session_state.current_day += 1
            st.session_state.foods = []

            st.success(f"Day {st.session_state.current_day - 1} has been closed. Moving to Day {st.session_state.current_day}.")

        # Remove details of previous days but keep cumulative nutrition
        for prev_day in range(1, st.session_state.current_day):
            if prev_day in st.session_state.cumulative_nutrition:
                st.write(f"---\n### Cumulative Nutrition for Day {prev_day}:")
                st.write(f"- **Total Calories:** {st.session_state.cumulative_nutrition[prev_day]['calories']:.2f} kcal")
                st.write(f"- **Total Protein:** {st.session_state.cumulative_nutrition[prev_day]['protein']:.2f} g")
                st.write(f"- **Total Carbs:** {st.session_state.cumulative_nutrition[prev_day]['carbs']:.2f} g")
                st.write(f"- **Total Fat:** {st.session_state.cumulative_nutrition[prev_day]['fat']:.2f} g")


def main():
    st.title('Nutrition Tracker App')

    # Select if the user is New or Existing
    user_type = st.radio('Are you a new or existing user?', ['New User', 'Existing User'])

    # --- For New Users ---
    if user_type == 'New User':
        st.header('Enter Your Details to Calculate TDEE')

        # Collecting input for TDEE calculation
        age = st.number_input('Age', min_value=1, max_value=120, value=25)

        # Gender selection
        gender = st.radio('Gender', options=['Male', 'Female'])
        gender_encoded = 1 if gender == 'Male' else 0

        # Height and Weight inputs
        height_cm = st.number_input('Height (in cm)', min_value=50.0, max_value=250.0, value=170.0)
        weight_kg = st.number_input('Weight (in kg)', min_value=20.0, max_value=300.0, value=70.0)

        # Active Hours (excluding workout)
        st.write('Active Hours (excluding workout)')
        active_hours_hr = st.number_input('Hours', min_value=0, max_value=24, value=1)
        active_hours_min = st.number_input('Minutes', min_value=0, max_value=59, value=0)
        active_hours = active_hours_hr + (active_hours_min / 60)

        # Workout Type and Hours
        workout_type = st.radio('Workout Type', options=['Cardio', 'Strength', 'Mixed', 'No Workout'])
        st.write('Workout Hours')
        workout_hours_hr = st.number_input('Hrs', min_value=0, max_value=24, value=0)
        workout_hours_min = st.number_input('Mins', min_value=0, max_value=59, value=0)
        workout_hours = workout_hours_hr + (workout_hours_min / 60)

        # Plan Type and weight change
        plan_type = st.radio('Plan Type', options=['Weight Loss', 'Weight Gain'])
        kg_change = st.number_input('Weight to Reduce/Gain per month (in kg)', min_value=0.0, max_value=10.0, value=2.0)
        st.write('Recommendation: 1-4 kg is recommended.')

        # BMI Calculation
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)

        # BMR Prediction
        bmr_input = np.array([[age, weight_kg, height_cm, gender_encoded]])
        predicted_bmr = bmr_model.predict(bmr_input)[0]

        # TDEE Calculation
        tdee = predicted_bmr

        # Calories Burned Prediction
        if workout_type == 'No Workout':
            noworkout_input = np.array([[age, gender_encoded, bmi, active_hours]])
            calories_burned = noworkout_model.predict(noworkout_input)[0]
        else:
            workout_type_dict = {'Cardio': 0, 'Mixed': 1, 'Strength': 2}
            workout_type_encoded = workout_type_dict[workout_type]
            workout_input = np.array([[age, gender_encoded, bmi, active_hours, workout_hours, workout_type_encoded]])
            calories_burned = workout_model.predict(workout_input)[0]

        # Total TDEE before adjustment
        tdee += calories_burned

        # Adjusting TDEE based on Plan Type
        calorie_adjustment = (7700 * kg_change) / 30
        if plan_type == 'Weight Loss':
            adjusted_tdee = tdee - calorie_adjustment
        else:
            adjusted_tdee = tdee + calorie_adjustment

        if st.button('Calculate TDEE'):
            st.subheader('Results')
            st.write(f'**Your BMR is:** {predicted_bmr:.2f} kcal/day')
            st.write(f'**Calories Burned from Activity:** {calories_burned:.2f} kcal/day')
            st.write(f'**Unadjusted TDEE:** {tdee:.2f} kcal/day')
            
            if plan_type == 'Weight Loss':
                st.write(f'**To lose {kg_change} kg per month, your adjusted TDEE is:** {adjusted_tdee:.2f} kcal/day')
                st.write('You should consume this amount or fewer calories per day.')
            else:
                st.write(f'**To gain {kg_change} kg per month, your adjusted TDEE is:** {adjusted_tdee:.2f} kcal/day')
                st.write('You should consume this amount or more calories per day.')

            # Store the calculated TDEE and plan type for tracker section
            st.session_state.adjusted_tdee = adjusted_tdee
            st.session_state.plan_type = plan_type

            # Insert the new user data into the database
            user_id = add_new_user(age, gender_encoded, height_cm, weight_kg, active_hours, workout_type, workout_hours, plan_type, kg_change, adjusted_tdee)
            st.success(f'User created successfully! Your User ID is {user_id}')
            
            # Store the user ID in the session state
            st.session_state.user_id = user_id
            st.session_state.current_day = 1  # New users start at Day 1

    # --- For Existing Users ---
    # --- For Existing Users ---
    elif user_type == 'Existing User':
        user_id = st.number_input('Enter your User ID', min_value=1)

        if st.button('Continue'):
        # Retrieve user data from the database
            user_data = get_user_data(user_id)

            if user_data:
            # Initialize session state variables for existing users
                st.session_state.user_id = user_id
                st.session_state.adjusted_tdee = user_data[10]  # Adjusted TDEE from DB
                st.session_state.plan_type = user_data[8]  # This should contain 'Weight Loss' or 'Weight Gain'

            # Fetch the last day the user logged food
                last_day = get_last_logged_day(user_id)
                st.session_state.current_day = last_day + 1  # Start from the next day after the last logged day

                st.success(f'Welcome back! You are starting from Day {st.session_state.current_day}')
               
            else:
                st.error('User not found! Please check your User ID.')

            

    # Show the tracker section only after TDEE calculation or if user exists
    if 'user_id' in st.session_state:
        tracker_section()

        if st.button("Show Inference"):
                st.write("### Dynamic Visuals")
                # Visualize user progress
                visualize_user_progress(user_id)

                # Display weight change based on caloric surplus/deficit
                calculate_weight_change(user_id)

        

if __name__ == '__main__':
    main()
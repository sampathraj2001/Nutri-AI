# 📱 Calorie Tracker and Nutrition Analysis App

This project is a **Calorie Tracker and Nutrition Analysis** system designed to help users track their daily food intake and receive feedback on whether their caloric consumption aligns with their fitness goals (weight loss or weight gain). The app uses machine learning models for **TDEE prediction** and **image-based food recognition**.

## ✨ Features

### 1. 🔥 TDEE Prediction:
   - Calculates **Total Daily Energy Expenditure (TDEE)** based on user data like height, weight, age, activity level, and workout type.
   - The TDEE is dynamically adjusted based on the user's fitness plan (weight loss or weight gain).

### 2. 🖼️ Image-Based Food Recognition:
   - Utilizes a **pre-trained ViT (Vision Transformer) model** from Hugging Face to recognize food from images.
   - Maps the predicted food to a pre-defined nutrition dataset (Indian foods) for calorie, protein, carbs, and fat data.

### 3. 📅 Daily Tracking:
   - Users can add food to different days, with expandable features to track food intake day by day.
   - Provides cumulative nutrition facts for each day and evaluates if the daily calorie intake meets the user's fitness goals.

### 4. 🎯 Goal-Based Feedback:
   - Provides feedback on whether the user’s daily calorie intake is aligned with their Adjusted TDEE based on their selected plan (weight loss or weight gain).


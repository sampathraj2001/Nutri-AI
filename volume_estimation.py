import cv2
import numpy as np

def estimate_food_volume(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    food_area_pixels = sum(cv2.contourArea(c) for c in contours)

    conversion_factor = 0.01  # Adjust this factor for your specific use case
    predicted_volume = food_area_pixels * conversion_factor

    return predicted_volume

if __name__ == "__main__":
    image_path = "pizza.jpg"  
    volume = estimate_food_volume(image_path)
    print(f"Predicted food volume: {volume} cubic centimeters")
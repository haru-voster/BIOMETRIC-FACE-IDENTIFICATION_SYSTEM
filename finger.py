import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import mysql.connector
import time

# Additional imports for fingerprint enrollment and processing
import fingerprint_enrollment as fp_enroll
import fingerprint_processing as fp_process

# Initialize Tkinter window
window = tk.Tk()
window.title("FAMS - Face and Fingerprint Recognition Based Attendance Management System")
window.geometry('1280x720')
window.configure(background='grey80')

# Function to enroll fingerprint for a single person
def enroll_fingerprint():
    # Capture fingerprint images
    fingerprint_images = fp_enroll.capture_fingerprint()

    if fingerprint_images is None:
        messagebox.showerror("Error", "Failed to capture fingerprint images")
        return

    # Process fingerprint images to extract minutiae
    minutiae = fp_process.extract_minutiae(fingerprint_images)

    if minutiae is None:
        messagebox.showerror("Error", "Failed to extract minutiae from fingerprint images")
        return

    # Store minutiae data in the database (for the single person)
    store_minutiae_in_database(minutiae)

    messagebox.showinfo("Success", "Fingerprint enrolled successfully")

# Function to store minutiae data in the database
def store_minutiae_in_database(minutiae):
    # Connect to the database
    try:
        connection = mysql.connector.connect(
            host='localhost', user='root', password='@voster', database='voster')
        cursor = connection.cursor()
    except Exception as e:
        messagebox.showerror("Error", f"Database connection error: {e}")
        return

    # Insert minutiae data into the database
    try:
        query = "INSERT INTO fingerprint (minutiae_data) VALUES (%s)"
        values = (minutiae,)
        cursor.execute(query, values)
        connection.commit()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to insert minutiae data into database: {e}")
        connection.rollback()
    finally:
        cursor.close()
        connection.close()

# Function to handle admission of students
def admit_student():
    # Add your code for admitting students here
    pass

# Function for admin check
def admin_check():
    # Add your admin authentication code here
    pass

# Buttons and GUI elements
enrollFingerprintButton = tk.Button(window, text="Enroll Fingerprint", command=enroll_fingerprint, 
                                     fg="black", bg="SkyBlue1", width=20, height=3, 
                                     activebackground="green", font=('times', 15, 'bold'))
enrollFingerprintButton.place(x=50, y=200)

admitStudentButton = tk.Button(window, text="Admit Student", command=admit_student, 
                               fg="black", bg="SkyBlue1", width=20, height=3, 
                               activebackground="green", font=('times', 15, 'bold'))
admitStudentButton.place(x=300, y=200)

adminCheckButton = tk.Button(window, text="Admin", command=admin_check, 
                             fg="black", bg="SkyBlue1", width=20, height=3, 
                             activebackground="green", font=('times', 15, 'bold'))
adminCheckButton.place(x=550, y=200)

window.mainloop()

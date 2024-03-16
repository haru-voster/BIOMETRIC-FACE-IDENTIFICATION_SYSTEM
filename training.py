import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import mysql.connector
import time

# Window is our Main frame of the system
window = tk.Tk()
window.title("FAMS - Face Recognition Based Attendance Management System")
window.geometry('1280x720')
window.configure(background='grey80')
def get_selected_date():
        selected_date = cal.get_date()
        print("Selected Date:", selected_date)
from tkcalendar import Calendar

def display_calendar():
            top = tk.Toplevel(root)

            cal = Calendar(top, selectmode="day", date_pattern="yy/mm/dd")
            cal.pack(fill="both", expand=True)

            select_button = tk.Button(top, text="Select Date", command=get_selected_date)
            select_button.pack()

root = tk.Tk()
root.title("Date Picker")

cal = Calendar(root, selectmode="none", date_pattern="yy/mm/dd")
cal.pack()

select_button = tk.Button(root, text="Select Date", command=display_calendar)
select_button.pack()


# Function to store information and images in the database
def voster(student_adm, student_firstname, student_lastname, date_joined, year, Drawing):
    # Connect to the database
    try:
        connection = mysql.connector.connect(
            host='localhost', user='root', password='@voster', database='voster')
        cursor = connection.cursor()
    except Exception as e:
        messagebox.showerror("Error", f"Database connection error: {e}")
        return
    # Inserting data into the database
    try:
        query = "INSERT INTO students (STUDENT_ADM, STUDENT_FIRSTNAME, STUDENT_LASTNAME, DATE_JOINED, YEAR, DRAWING) VALUES (%s, %s, %s, %s, %s, %s)"
        values = (student_adm, student_firstname, student_lastname, date_joined, year, Drawing)
        cursor.execute(query, values)
        connection.commit()
        messagebox.showinfo("Success", "Student information saved successfully")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to insert data into database: {e}")
        connection.rollback()
    finally:
        cursor.close()
        connection.close()

# Function to capture image and recognize face
def recognize_face(student_adm, student_firstname, img):
    try:
        detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("TrainingImageModel/Trainner.yml")
        font = cv2.FONT_HERSHEY_SIMPLEX
        id = 0

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if confidence < 50:
                cursor.execute(f"SELECT STUDENT_ADM, STUDENT_FIRSTNAME FROM students WHERE STUDENT_ADM = {id}")
                result = cursor.fetchone()
                id = result[0]
                confidence = f"{100 - round(confidence)}%"
                cv2.putText(img, str(result[1]), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            else:
                id = "Unknown"
                confidence = f"{100 - round(confidence)}%"
                cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)

        cv2.imshow('Face Recognition', img)
        cv2.waitKey(400)  # Display the image for 500 milliseconds
        cv2.destroyAllWindows()
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# For take images for datasets
# Function to convert image data to Base64 string
def convert_image_to_drawing(img_path):
    with open(img_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read())
    return encoded_string.decode('utf-8')
import base64#dfor images storage
def take_img():
    # Function to save images and store information in the database
    def capture_and_save_image():
        student_adm = txt.get()
        student_firstname = txt2.get()
        student_lastname = txt3.get()
        date_joined = txt4.get()
        year = txt5.get()

        if student_adm == '' or student_firstname == '' or student_lastname == '' or date_joined =='' or year =='':
            messagebox.showerror("Error", "Please fill all the fields")
            return

        try:
            # Open camera
            cam = cv2.VideoCapture(0)
            detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            sampleNum = 0
            while True:
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                    # Incrementing sample number
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    # Incrementing sample number
                    sampleNum += 1
                    # Saving the captured face in the dataset folder
                    cv2.imwrite(f"TrainingImage/{student_adm}.{sampleNum}.jpg", gray[y:y + h, x:x + w])

                if sampleNum > 1:
                    break
                # Read the saved image file as binary data
           # Convert image to Base64 string
            image_path = f"TrainingImage/{student_adm}.1.jpg"
            drawing = convert_image_to_drawing(image_path)
    # Connect to the database
            connection = mysql.connector.connect(
                host='localhost', user='root', password='@voster', database='voster')
            cursor = connection.cursor()

            # Inserting data into the database
            query = "INSERT INTO students (STUDENT_ADM, STUDENT_FIRSTNAME, STUDENT_LASTNAME, DATE_JOINED, YEAR, DRAWING) VALUES (%s, %s, %s, %s, %s, %s)"
            values = (student_adm, student_firstname, student_lastname, date_joined, year, drawing)
            cursor.execute(query, values)
            connection.commit()
            messagebox.showinfo("Success", "Student information saved successfully")

            cursor.close()
            connection.close()

            # Recognize face
            recognize_face(student_adm, student_firstname, drawing)

            cam.release()
            cv2.destroyAllWindows()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

# Buttons and other GUI elements 
#clearing
    def remove_enr():
        ENR_ENTRY.delete(first=0, last=22)

        STUDENT_ENTRY = tk.Entry(
        MFW, width=20, bg="white", fg="black", font=('times', 23))
        STUDENT_ENTRY.place(x=290, y=205)
#this happens inside the registration of new student

    # Create a new window for taking images
    img_window = tk.Toplevel(window)
    img_window.title("Take Images")
    img_window.geometry('500x700')
    img_window.configure(background='grey80')


    lbl = tk.Label(img_window, text="ADMISSION NUMBER:", width=20, fg="black", bg="grey", height=2, font=('times', 15, 'bold'))
    lbl.place(x=50, y=50)
    txt = tk.Entry(img_window, width=20, bg="white", fg="black", font=('times', 25))
    txt.place(x=290, y=60)

    lbl2 = tk.Label(img_window, text="FIRST NAME:", width=20, fg="black", bg="grey", height=2, font=('times', 15, 'bold'))
    lbl2.place(x=50, y=150)
    txt2 = tk.Entry(img_window, width=20, bg="white", fg="black", font=('times', 25))
    txt2.place(x=290, y=160)

    lbl3 = tk.Label(img_window, text="LAST NAME:", width=20, fg="black", bg="grey", height=2, font=('times', 15, 'bold'))
    lbl3.place(x=50, y=250)
    txt3 = tk.Entry(img_window, width=20, bg="white", fg="black", font=('times', 25))
    txt3.place(x=290, y=260)

    lbl4 = tk.Label(img_window, text="DATE JOINED:", width=20, fg="black", bg="grey", height=2, font=('times', 15, 'bold'))
    lbl4.place(x=50, y=350)
    txt4 = tk.Entry(img_window, width=20, bg="white", fg="black", font=('times', 25))
    txt4.place(x=290, y=360)

    lbl5 = tk.Label(img_window, text="YEAR OF STUDY:", width=20, fg="black", bg="grey", height=2, font=('times', 15, 'bold'))
    lbl5.place(x=50, y=450)
    txt5 = tk.Entry(img_window, width=20, bg="white", fg="black", font=('times', 25))
    txt5.place(x=290, y=460)
    #text clearance in execution
    
    save_button = tk.Button(img_window, text="Save", command=capture_and_save_image, fg="black", bg="SkyBlue1", width=10, height=1, activebackground="white", font=('times', 15, 'bold'))
    save_button.place(x=200, y=550)

# Buttons and other GUI elements
message = tk.Label(window, text="Students-Face-Recognition-Based-Attendance-Management-System", bg="black", fg="white", width=50, height=3, font=('times', 30, ' bold '))
takeImg = tk.Button(window, text="ADMIT STUDENTS", command=take_img, fg="black", bg="SkyBlue1", width=20, height=3, activebackground="green", font=('times', 15, 'bold'))
takeImg.place(x=50, y=400)
#FUNCTION TRAIN IMAGE
import os

def train_model():
    # Path to the dataset containing images of students
    dataset_path = "TrainingImage/"

    # Initialize lists to store face samples and corresponding labels
    faces = []
    labels = []

    # Create a face recognizer object (LBPH)
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Get the paths of all image files in the dataset
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]

    # Iterate through each image file
    for image_path in image_paths:
        # Read the image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Extract the label from the filename (format: student_adm.sampleNum.jpg)
        label = int(os.path.split(image_path)[-1].split(".")[0])

        # Detect faces in the image
        faces_detected = face_detector.detectMultiScale(img)

        # Iterate through each detected face
        for (x, y, w, h) in faces_detected:
            # Extract the face region
            face = img[y:y+h, x:x+w]
            # Append the face sample and corresponding label to lists
            faces.append(face)
            labels.append(label)

    # Train the face recognizer using the collected samples and labels
    recognizer.train(faces, np.array(labels))

    # Save the trained model to a file
    recognizer.save("TrainingImageModel/Trainner.yml")

    print("Training completed successfully.")

# Load the face detection cascade classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#attendance
def mark_attendance():
    def save_attendance():
        student_adm = txt.get()
        student_firstname = txt2.get()
        mark_attendance_to_db(student_adm, student_firstname)

    def mark_attendance_to_db(student_adm, student_firstname):
        if student_adm == '' or student_firstname == '':
            messagebox.showerror("Error", "Please fill all the fields")
            return

        try:
            # Connect to MySQL database
            connection = mysql.connector.connect(
                host='localhost',  # Specify your host
                user='root',  # Specify your username
                password='@voster',  # Specify your password
                database='voster'  # Specify your database name
            )
            cursor = connection.cursor()

            # Check if the student is registered
            query = "SELECT * FROM students WHERE STUDENT_ADM = %s AND STUDENT_FIRSTNAME = %s"
            values = (student_adm, student_firstname)
            cursor.execute(query, values)
            result = cursor.fetchone()

            if result is None:
                messagebox.showerror("Error", "Student is not registered")
                return

            # Open camera and detect faces
            cam = cv2.VideoCapture(0)
            detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            sampleNum = 0
            
            # Capture student images
            while True:
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    sampleNum += 1
                    image_path = f"TrainingImage/{student_adm}.{sampleNum}.jpg"
                    cv2.imwrite(image_path, gray[y:y + h, x:x + w])
                
                if sampleNum > 1:
                    break

            # Insert attendance record into the database
            current_date = time.strftime('%Y-%m-%d')
            current_time = time.strftime('%H:%M:%S')
            query = "INSERT INTO attendance (STUDENT_ADM, STUDENT_FIRSTNAME, DATE, TIME) VALUES (%s, %s, %s, %s)"
            values = (student_adm, student_firstname, current_date, current_time)
            cursor.execute(query, values)
            connection.commit()
            messagebox.showinfo("Success", "Attendance marked successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to mark attendance: {e}")

        finally:
            # Close connection and release camera
            if 'connection' in locals() and connection.is_connected():
                cursor.close()
                connection.close()
            cam.release()

    # Create a new window for marking attendance
    att_window = tk.Toplevel(window) 
    att_window.title("Mark Attendance")
    att_window.geometry('500x400')
    att_window.configure(background='grey80')

    lbl = tk.Label(att_window, text="ADMISSION NUMBER:", width=20, fg="black", bg="grey", height=2, font=('times', 15, 'bold'))
    lbl.place(x=50, y=50)
    txt = tk.Entry(att_window, width=20, bg="white", fg="black", font=('times', 25))
    txt.place(x=290, y=60)

    lbl2 = tk.Label(att_window, text="FIRST NAME:", width=20, fg="black", bg="grey", height=2, font=('times', 15, 'bold'))
    lbl2.place(x=50, y=150)
    txt2 = tk.Entry(att_window, width=20, bg="white", fg="black", font=('times', 25))
    txt2.place(x=290, y=160)

    save_button = tk.Button(att_window, text="ENROL FACE", command=save_attendance, fg="black", bg="SkyBlue1", width=10, height=1, activebackground="white", font=('times', 15, 'bold'))
    save_button.place(x=200, y=250)
#
# Function for admin checking
def admin_check():
 # Implement your admin checking logic here
    win = tk.Tk()
    # win.iconbitmap('AMS.ico')
    win.title("LogIn")
    win.geometry('880x420')
    win.configure(background='grey80')

    def log_in():
        username = un_entr.get()
        password = pw_entr.get()

        if username == 'VOSTER':  # Changed admin
            if password == 'voster123':  # Admin password
                win.destroy()
                import csv
                import tkinter

                root = tkinter.Tk()
                root.title("Student Details")
                root.configure(background='grey80')

                try:
                    connection = mysql.connector.connect(
                        host='localhost', user='root', password='@voster', database='voster')
                    cursor = connection.cursor(dictionary=True)

                    cursor.execute("SELECT * FROM students")
                    students = cursor.fetchall()

                    for idx, student in enumerate(students):
                        for col_idx, (key, value) in enumerate(student.items()):
                            label_text = f"{key}: {value}"
                            label = tkinter.Label(root, width=30, height=1, fg="black",
                                                    font=('times', 12), bg="white",
                                                    text=label_text, anchor="w", relief=tkinter.RIDGE)
                            label.grid(row=idx, column=col_idx, sticky="ew")

                        cursor.close()
                        connection.close()

                except mysql.connector.Error as e:
                    messagebox.showerror("Error", f"Database error: {e}")

                root.mainloop()
            else:
                valid = 'Incorrect Username or Password'
                Nt.configure(text=valid, bg="red", fg="white",
                             width=38, font=('times', 19, 'bold'))
                Nt.place(x=120, y=350)

        else:
            valid = 'Incorrect Username or Password'
            Nt.configure(text=valid, bg="red", fg="white",
                         width=38, font=('times', 19, 'bold'))
            Nt.place(x=120, y=350)


    Nt = tk.Label(win, text="Attendance filled Successfully", bg="Green", fg="white", width=40,
                  height=2, font=('times', 19, 'bold'))
    # Nt.place(x=120, y=350)

    un = tk.Label(win, text="Enter username : ", width=15, height=2, fg="black", bg="grey",
                  font=('times', 15, ' bold '))
    un.place(x=30, y=50)

    pw = tk.Label(win, text="Enter password : ", width=15, height=2, fg="black", bg="grey",
                  font=('times', 15, ' bold '))
    pw.place(x=30, y=150)

    def c00():
        un_entr.delete(first=0, last=22)

    un_entr = tk.Entry(win, width=20, bg="white", fg="black",
                       font=('times', 23))
    un_entr.place(x=290, y=55)

    def c11():
        pw_entr.delete(first=0, last=22)

    pw_entr = tk.Entry(win, width=20, show="*", bg="white",
                       fg="black", font=('times', 23))
    pw_entr.place(x=290, y=155)

    c0 = tk.Button(win, text="Clear", command=c00, fg="white", bg="black", width=10, height=1,
                   activebackground="white", font=('times', 15, ' bold '))
    c0.place(x=690, y=55)

    c1 = tk.Button(win, text="Clear", command=c11, fg="white", bg="black", width=10, height=1,
                   activebackground="white", font=('times', 15, ' bold '))
    c1.place(x=690, y=155)

    Login = tk.Button(win, text="LogIn", fg="black", bg="SkyBlue1", width=20,
                      height=2,
                      activebackground="Red", command=log_in, font=('times', 15, ' bold '))
    Login.place(x=290, y=250)
    win.mainloop()

message = tk.Label(window, text="STUDENTS FACE BASED ATTENDANCE SYSTEM", bg="black", fg="white", width=50,
                   height=4, font=('times', 30, ' bold '))

message.place(x=80, y=20)
#TRAINING THE IMAGE
train_model_button = tk.Button(window, text="TRAIN MODEL", fg="black", command=train_model, bg="SkyBlue1",
                     width=20, height=2, activebackground="white", font=('times', 15, ' bold '))
train_model_button.place(x=550, y=300)
# Button for automatic attendance
autoAttendance = tk.Button(window, text="ATTENDANCE", command=mark_attendance, bg="SkyBlue1", fg="black", 
                           width=20, height=2, activebackground="green", font=('times', 15, 'bold'))
autoAttendance.place(x=300, y=300)

adminCheck = tk.Button(window, text="ADMIN", command=admin_check, fg="black", bg="SkyBlue1", 
                       width=20, height=2, activebackground="green", font=('times', 15, 'bold'))
adminCheck.place(x=800, y=400)
message = tk.Label(window,  text= "COMMITTED TO INNOVATION AND EXCELLENCE")

window.mainloop()
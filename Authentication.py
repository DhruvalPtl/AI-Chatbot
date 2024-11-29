import streamlit as st
import firebase_admin
from firebase_admin import credentials
import pyrebase
import json

# Initialize Firebase app if not already initialized
if not firebase_admin._apps:
    try:
        firebase_config = dict(st.secrets["firebaseapp"])
        cred = credentials.Certificate(firebase_config)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"Failed to initialize Firebase: {e}")

firebase = pyrebase.initialize_app(st.secrets["firebaseconfig"])
authentication = firebase.auth()

class Database:
    def __init__(self,user_id):
        self.db = firebase.database()
        self.user_id = user_id

    def chat_history(self):
        try:
            chat_history = self.db.child("users").child(self.user_id).child("chat_history").get().val()
                
            if chat_history is None:
                return []

            # Assuming chat_history is a dictionary, convert it to a list of dictionaries or a desired structure
            # Here, we're assuming it's a list-like structure where each message is stored as a dictionary
            return chat_history
        
        except Exception as e:
            # Handle exceptions (e.g., network issues, database errors)
            print(f"Error fetching chat history: {e}")
            return []  # Return an empty list on error

    def save_chat_to_database(self,user_id, messages):
        """Save chat history to Firebase Realtime Database."""
        try:
            # Prepare data for database
            chat_data = [
                {
                    "role": msg.role,
                    "parts": [part.text for part in msg.parts]
                }
                for msg in messages
            ]
            # Write data to the user's chat history in Firebase
            self.db.child("users").child(user_id).child("chat_history").set(chat_data)
        except Exception as e:
            st.error(f"Failed to save chat history: {e}")

class Authentication:
    def __init__(self):
        self.db = firebase.database()
        
    def sign_up(self,user_name,email,password):
        try:
            user = authentication.create_user_with_email_and_password(
                email=email, 
                password=password
            )
            user_id = user["localId"]
            
            user_data = {
                "user_name": user_name,
                "email": email,
            }
            self.db.child("users").child(user_id).set(user_data)

            st.success("Sign Up successfully!")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    def login(self,email,password):
        try:
            user = authentication.sign_in_with_email_and_password(
                email=email,
                password=password
            )
            
            if user:
                st.success("Login successful!")
            
            user_id = user["localId"]

            user_data = self.db.child("users").child(user_id).get().val()
            
            if user_data:
                st.session_state["user_id"] = user_id
                st.session_state["user_name"] = user_data["user_name"]
                st.success(f"Welcome back, {user_data['user_name']}!")
            else:
                st.error("User data not found.")
                        
        except Exception as e:
            # st.error(f"An error occurred: {e}")
            st.error("No account found with this email address and password. Please check your email or password.")
        
def logout():
    st.session_state.clear()
    st.success("Logged out successfully!")

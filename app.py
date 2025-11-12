from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response
from functools import wraps
from urllib.parse import urlparse, urljoin
import json
from datetime import datetime, timedelta
import pandas as pd
import pytz
from pymongo import MongoClient
import pymongo
from bson import ObjectId
import os
import requests  # ✅ ADD THIS IMPORT
from werkzeug.utils import secure_filename  # ✅ ADD THIS IMPORT
import mimetypes
app = Flask(__name__)
app.secret_key = 'your-secret-key'  # already present
app.permanent_session_lifetime = timedelta(days=1)

# Azure backend base URL - Global configuration
BASE_URL = "https://capps-backend-ooxig22w5exq6.calmpebble-4f694259.westus.azurecontainerapps.io"


def is_safe_url(target: str) -> bool:
    """
    Ensure the target URL is safe to redirect to (same host and http/https scheme)
    """
    if not target:
        return False
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return (
        test_url.scheme in ("http", "https")
        and ref_url.netloc == test_url.netloc
    )


def login_required(view_func):
    """
    Decorator to enforce authentication for view functions.
    Redirects unauthenticated users to the login page and preserves the
    original destination (when safe) using the `next` query parameter.
    """
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if 'email' not in session:
            next_url = request.full_path if request.method in ('GET', 'HEAD') else ''
            if not is_safe_url(next_url):
                next_url = ''
            return redirect(url_for('login', next=next_url or None))
        return view_func(*args, **kwargs)
    return wrapped_view

# MongoDB connection setup
def get_mongo_connection(collection_name='VincentBotReplicaWithVoice'):
    mongo_uri = "mongodb+srv://userawais1:awais645@cluster0.fcqph.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(mongo_uri)
    db = client["chat_history"]
    collection = db[collection_name]
    return collection

def read_chat_data(user_email=None, user_role=None):
    try:
        print("Reading data from MongoDB collection...")
        
        # Get data from VincentBotReplicaWithVoice collection only
        collection = get_mongo_connection()
        
        # Create query filter based on user email and role
        query_filter = {}
        if user_role == 'member' and user_email:
            # If user is member, only show their own data
            query_filter['email'] = user_email
            print(f"Member user - Filtering data for user: {user_email}")
        elif user_role == 'admin':
            # If user is admin, show all data (no filter)
            print("Admin user - Showing all data")
        else:
            # Default to member behavior if role is not specified
            query_filter['email'] = user_email
            print(f"Default behavior - Filtering data for user: {user_email}")
        
        cursor = collection.find(query_filter).sort("timestamp", pymongo.DESCENDING)
        df = pd.DataFrame(list(cursor))
        print(f"Retrieved {len(df)} records from collection")
        
        # Check if dataframe is empty
        if df.empty:
            print("No data found in MongoDB collection")
            return None
        
        # Field mapping based on your document structure
        field_mapping = {
            'email': 'Email',
            'user_query': 'Question',      # ✅ This is correct
            'bot_response': 'Answer', 
            'timestamp': 'Date',
            'feedback': 'Feedback'
        }
        
        # Apply field mappings
        for mongo_field, expected_field in field_mapping.items():
            if mongo_field in df.columns:
                df[expected_field] = df[mongo_field]
                print(f"Mapped {mongo_field} to {expected_field}")
        
        # Handle timestamp field (keep your existing timestamp processing code)
        if 'timestamp' in df.columns:
            print("Converting timestamp to datetime...")
            
            def convert_timestamp(ts):
                if isinstance(ts, dict) and '$date' in ts:
                    # Handle MongoDB $date format
                    if '$numberLong' in ts['$date']:
                        ts_long = int(ts['$date']['$numberLong'])
                        return datetime.fromtimestamp(ts_long / 1000, tz=pytz.UTC)
                    else:
                        # Handle other date formats if needed
                        return pd.NaT
                else:
                    # Try to parse as is
                    try:
                        return pd.to_datetime(ts)
                    except:
                        return pd.NaT
            
            # Apply the conversion function
            df['Date'] = df['timestamp'].apply(convert_timestamp)
            print(f"Sample converted dates: {df['Date'].head(3)}")
        elif 'Date' not in df.columns:
            # Use ObjectId generation time as fallback
            print("No timestamp found, using _id to generate dates...")
            df['Date'] = df['_id'].apply(lambda x: x.generation_time if isinstance(x, ObjectId) else pd.NaT)
        
        # Make sure Date column is timezone aware
        if 'Date' in df.columns and not df['Date'].empty:
            # Ensure all dates are timezone aware
            if not pd.api.types.is_datetime64tz_dtype(df['Date']):
                df['Date'] = df['Date'].dt.tz_localize(pytz.UTC)
            
            print(f"Date column dtype: {df['Date'].dtype}")
            print(f"Sample dates: {df['Date'].head(3)}")
        
        # Sort by date
        df = df.sort_values(by='Date', ascending=False)
        
        return df
    except Exception as e:
        print(f"Error reading MongoDB data: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_data_by_period(period='today', start_date=None, end_date=None, user_filter=None, user_email=None, user_role=None):
    df = read_chat_data(user_email=user_email, user_role=user_role)
    if df is None:
        # Return empty data structure
        return {
            "labels": [],
            "values": [],
            "total_messages": 0,
            "tickets_created": 0,
            "user_percentages": [],
            "feedback_stats": {"Good": 0, "Bad": 0, "Neutral": 0}
        }
    
    # Make sure we're using timezone-aware datetime objects for comparison
    current_time = datetime.now(pytz.UTC)
    
    # Filter data based on selected period
    if period == 'custom' and start_date and end_date:
        try:
            start_date = datetime.strptime(start_date, "%Y-%m-%d").replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59, microsecond=999)
            # Make timezone aware
            start_date = pytz.UTC.localize(start_date)
            end_date = pytz.UTC.localize(end_date)
            df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        except ValueError:
            # If date parsing fails, fallback to today
            start_date = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = current_time
            df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    elif period == 'today':
        start_date = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = current_time
        df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    elif period == 'yesterday':
        start_date = (current_time - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=1) - timedelta(microseconds=1)
        df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    elif period == 'this-week':
        start_date = (current_time - timedelta(days=current_time.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        df_filtered = df[df['Date'] >= start_date]
    elif period == 'last-week':
        start_date = (current_time - timedelta(days=current_time.weekday() + 7)).replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=7) - timedelta(microseconds=1)
        df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    elif period == 'this-month':
        start_date = current_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        df_filtered = df[df['Date'] >= start_date]
    elif period == 'last-month':
        last_month = current_time.month - 1 if current_time.month > 1 else 12
        last_month_year = current_time.year if current_time.month > 1 else current_time.year - 1
        start_date = current_time.replace(year=last_month_year, month=last_month, day=1, hour=0, minute=0, second=0, microsecond=0)
        end_date = current_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0) - timedelta(microseconds=1)
        df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    else:
        # Default to all data
        df_filtered = df
    
    # Apply user filter if provided (additional filtering for admin users)
    if user_filter and isinstance(user_filter, list) and len(user_filter) > 0 and 'Email' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Email'].isin(user_filter)]
        print(f"Applied additional user filter: {user_filter}. Remaining messages: {len(df_filtered)}")
    
    # Calculate feedback statistics
    feedback_stats = {"Good": 0, "Bad": 0, "Neutral": 0}
    if 'Feedback' in df_filtered.columns:
        feedback_counts = df_filtered['Feedback'].value_counts()
        for feedback, count in feedback_counts.items():
            if feedback in feedback_stats:
                feedback_stats[feedback] = int(count)
            elif feedback == "":
                feedback_stats["Neutral"] += int(count)
    
    # Group by appropriate time interval based on the period
    if period in ['today', 'yesterday']:
        # Group by hour for single day views
        df_filtered['Hour'] = df_filtered['Date'].dt.hour
        df_filtered['Minute'] = df_filtered['Date'].dt.minute
        
        if not df_filtered.empty:
            # Get the latest time from the filtered data
            latest_record = df_filtered.loc[df_filtered['Date'].idxmax()]
            latest_hour = latest_record['Hour']
            latest_minute = latest_record['Minute']
            latest_time_str = f"{int(latest_hour):02d}:{int(latest_minute):02d}"
            
            # Group by hour
            hourly_counts = df_filtered.groupby('Hour').size().reset_index(name='count')
            
            # Generate a complete set of hours (0-latest_hour) with counts
            all_hours = pd.DataFrame({'Hour': range(latest_hour + 1)})
            hourly_counts = pd.merge(all_hours, hourly_counts, on='Hour', how='left').fillna(0)
            
            # Format labels as HH:00, with the last one showing the exact time
            labels = [f"{int(hour):02d}:00" for hour in hourly_counts['Hour']]
            if labels:
                labels[-1] = latest_time_str  # Replace the last label with the exact time
        else:
            # If no data, show empty hours
            labels = []
            values = []
            return {
                "labels": labels,
                "values": values,
                "total_messages": 0,
                "tickets_created": 0,
                "user_percentages": [],
                "feedback_stats": feedback_stats
            }
        
        values = hourly_counts['count'].astype(int).tolist()
    else:
        # Group by date for multi-day views
        df_filtered['Date_Day'] = df_filtered['Date'].dt.date
        
        if not df_filtered.empty:
            daily_counts = df_filtered.groupby('Date_Day').size().reset_index(name='count')
        
            # Prepare data for chart
            labels = [str(date) for date in daily_counts['Date_Day']]
            values = daily_counts['count'].tolist()
        else:
            # If no data, return empty arrays
            labels = []
            values = []
            return {
                "labels": labels,
                "values": values,
                "total_messages": 0,
                "tickets_created": 0,
                "user_percentages": [],
                "feedback_stats": feedback_stats
            }
    
    # Calculate metrics
    total_messages = len(df_filtered)
    # Count unique emails that created tickets (assume a message creates a ticket)
    tickets_created = df_filtered['Email'].nunique() if 'Email' in df_filtered.columns else 0
    
    # Calculate user chat percentages for donut chart
    user_percentages = []
    if not df_filtered.empty and 'Email' in df_filtered.columns:
        try:
            print("Calculating user percentages...")
            # Clean data
            df_filtered_clean = df_filtered.dropna(subset=['Email'])
            df_filtered_clean = df_filtered_clean[df_filtered_clean['Email'].str.strip() != '']
            
            if len(df_filtered_clean) > 0:
                user_counts = df_filtered_clean.groupby('Email').size().reset_index(name='count')
                print("User counts:", user_counts)
                
                total_messages = len(df_filtered_clean)
                print(f"Total messages: {total_messages}")
                
                if total_messages > 0:
                    user_counts['percentage'] = (user_counts['count'] / total_messages * 100).round(1)
                    user_counts = user_counts.sort_values('count', ascending=False)
                    
                    # Create list of user data for the donut chart
                    for _, row in user_counts.iterrows():
                        email = row['Email']
                        if email and isinstance(email, str):
                            user_percentages.append({
                                "username": email,
                                "count": int(row['count']),
                                "percentage": float(row['percentage'])
                            })
        except Exception as e:
            print(f"Error calculating user percentages: {e}")
            import traceback
            traceback.print_exc()
    
    # Prepare data for response
    chat_data = {
        "labels": labels,
        "values": values,
        "total_messages": total_messages,
        "tickets_created": tickets_created,
        "user_percentages": user_percentages,
        "feedback_stats": feedback_stats
    }
    
    print(f"Final response data structure: {chat_data}")
    return chat_data

@app.route('/')
def dashboard():
    if 'email' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

# Login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    next_url = request.args.get('next', '')

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        next_url = request.form.get('next', next_url)

        users_collection = get_mongo_connection('users')
        user = users_collection.find_one({'email': email, 'password': password})

        if user:
            session.permanent = True  
            session['email'] = email
            session['role'] = user.get('role', 'member')  # Get role from database, default to 'member'
            print(f"User {email} logged in with role: {session['role']}")
            if next_url and is_safe_url(next_url):
                return redirect(next_url)
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid credentials', next=next_url)

    return render_template('login.html', next=next_url)

@app.route('/logout')
def logout():
    session.pop('email', None)  # Remove user from session
    session.pop('role', None)   # Remove role from session
    return redirect(url_for('login'))  # Redirect to login page

@app.route('/api/chat_data')
def get_chat_data():
    # Check if user is authenticated
    if 'email' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    period = request.args.get('period', 'today')
    start_date = request.args.get('start_date', None)
    end_date = request.args.get('end_date', None)
    
    # Get user filter if provided
    user_filter = None
    users_param = request.args.get('users', None)
    if users_param:
        try:
            user_filter = json.loads(users_param)
            print(f"Received user filter: {user_filter}")
        except json.JSONDecodeError:
            print(f"Invalid JSON in users parameter: {users_param}")
    
    # Pass the authenticated user's email and role to filter data
    authenticated_email = session['email']
    user_role = session.get('role', 'member')
    
    data = get_data_by_period(period, start_date, end_date, user_filter, user_email=authenticated_email, user_role=user_role)
    return jsonify(data)

@app.route('/data')
def get_filtered_data():
    # Check if user is authenticated
    if 'email' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    start_date = request.args.get('start', None)
    end_date = request.args.get('end', None)
    
    # Get user filter if provided
    user_filter = None
    users_param = request.args.get('users', None)
    if users_param:
        try:
            user_filter = json.loads(users_param)
            print(f"Received user filter for /data route: {user_filter}")
        except json.JSONDecodeError:
            print(f"Invalid JSON in users parameter: {users_param}")
    
    if not start_date or not end_date:
        # If no date range is provided, use today
        current_time = datetime.now()
        start_date = current_time.strftime("%Y-%m-%d")
        end_date = current_time.strftime("%Y-%m-%d")
    
    # Get the filtered data with authenticated user's email and role
    authenticated_email = session['email']
    user_role = session.get('role', 'member')
    
    data = get_data_by_period('custom', start_date, end_date, user_filter, user_email=authenticated_email, user_role=user_role)
    
    # Format the response in the structure expected by the frontend
    response = {
        "linechart_data": {
            "labels": data["labels"],
            "values": data["values"]
        },
        "total_messages": data["total_messages"],
        "active_users": len(data["user_percentages"]) if data["user_percentages"] else 0,
        "user_data": data["user_percentages"],
        "feedback_stats": data["feedback_stats"],
        "period": "custom"
    }
    
    return jsonify(response)

@app.route('/chats')
@login_required
def chats():
    return render_template('chats.html')

@app.route('/api/chats')
def get_chat_list():
    # Check if user is authenticated
    if 'email' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    # Get authenticated user's email and role
    authenticated_email = session['email']
    user_role = session.get('role', 'member')
    print(f"Authenticated user: {authenticated_email}, Role: {user_role}")
    df = read_chat_data(user_email=authenticated_email, user_role=user_role)
    if df is None:
        return jsonify({
            "chats": [],
            "total_count": 0
        })
    
    # Get filter parameters
    period = request.args.get('period')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    username = request.args.get('username')
    users_param = request.args.get('users')
    feedback = request.args.get('feedback')
    
    # Debug: Print the filter parameters received
    print(f"Filter parameters: period={period}, start_date={start_date}, end_date={end_date}, users={users_param}")
    print(f"User role: {user_role}")
    
    # Apply date filters - using timezone-aware dates
    current_time = datetime.now(pytz.UTC)
    
    # Apply proper date filtering based on period
    if period:
        print(f"Applying period filter: {period}")
        if period == 'today':
            start_date_obj = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date_obj = current_time
            df = df[(df['Date'] >= start_date_obj) & (df['Date'] <= end_date_obj)]
            print(f"Today filter: {start_date_obj} to {end_date_obj}, records: {len(df)}")
        elif period == 'yesterday':
            start_date_obj = (current_time - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            end_date_obj = start_date_obj + timedelta(days=1) - timedelta(microseconds=1)
            df = df[(df['Date'] >= start_date_obj) & (df['Date'] <= end_date_obj)]
            print(f"Yesterday filter: {start_date_obj} to {end_date_obj}, records: {len(df)}")
        elif period == 'this-week':
            start_date_obj = (current_time - timedelta(days=current_time.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
            df = df[df['Date'] >= start_date_obj]
            print(f"This week filter: from {start_date_obj}, records: {len(df)}")
        elif period == 'last-week':
            start_date_obj = (current_time - timedelta(days=current_time.weekday() + 7)).replace(hour=0, minute=0, second=0, microsecond=0)
            end_date_obj = start_date_obj + timedelta(days=7) - timedelta(microseconds=1)
            df = df[(df['Date'] >= start_date_obj) & (df['Date'] <= end_date_obj)]
            print(f"Last week filter: {start_date_obj} to {end_date_obj}, records: {len(df)}")
        elif period == 'this-month':
            start_date_obj = current_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            df = df[df['Date'] >= start_date_obj]
            print(f"This month filter: from {start_date_obj}, records: {len(df)}")
        elif period == 'last-month':
            last_month = current_time.month - 1 if current_time.month > 1 else 12
            last_month_year = current_time.year if current_time.month > 1 else current_time.year - 1
            start_date_obj = current_time.replace(year=last_month_year, month=last_month, day=1, hour=0, minute=0, second=0, microsecond=0)
            end_date_obj = current_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0) - timedelta(microseconds=1)
            df = df[(df['Date'] >= start_date_obj) & (df['Date'] <= end_date_obj)]
            print(f"Last month filter: {start_date_obj} to {end_date_obj}, records: {len(df)}")
    elif start_date and end_date:
        try:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").replace(hour=0, minute=0, second=0, microsecond=0)
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59, microsecond=999)
            # Make timezone aware
            start_date_obj = pytz.UTC.localize(start_date_obj)
            end_date_obj = pytz.UTC.localize(end_date_obj)
            df = df[(df['Date'] >= start_date_obj) & (df['Date'] <= end_date_obj)]
            print(f"Custom date filter: {start_date_obj} to {end_date_obj}, records: {len(df)}")
        except ValueError as e:
            print(f"Date parsing error: {e}")
            # If date parsing fails, don't apply date filter

    # Apply username filter (only for admin users)
    if user_role == 'admin' and users_param:
        try:
            # Try to parse as JSON
            users = json.loads(users_param)
            print(f"Admin user - Parsed users_param: {users}")
            
            # Handle various formats that might come from frontend
            if isinstance(users, list):
                # Flatten nested lists
                flat_users = []
                for item in users:
                    if isinstance(item, list):
                        flat_users.extend([u for u in item if u and not pd.isna(u)])
                    elif item and not pd.isna(item):
                        flat_users.append(item)
                
                # Apply the filter if we have valid users
                if flat_users and 'Email' in df.columns:
                    df = df[df['Email'].isin(flat_users)]
                    print(f"Applied admin user filter with {len(flat_users)} users, remaining records: {len(df)}")
            elif isinstance(users, str) and users and 'Email' in df.columns:
                df = df[df['Email'] == users]
                print(f"Applied single admin user filter: {users}, remaining records: {len(df)}")
        except json.JSONDecodeError as e:
            print(f"Error parsing users JSON: {e}")
    elif user_role == 'admin' and username and 'Email' in df.columns:
        # Fallback to direct username filter for admin
        df = df[df['Email'].str.contains(str(username), case=False, na=False)]
        print(f"Applied admin username text search: {username}, remaining records: {len(df)}")
    
    # Apply feedback filter if provided
    if feedback and 'Feedback' in df.columns:
        df = df[df['Feedback'] == feedback]
        print(f"Applied feedback filter: {feedback}, remaining records: {len(df)}")
    
    # Sort by date (most recent first)
    df = df.sort_values(by='Date', ascending=False)
    
    # Format data for JSON response
    chats = []
    for _, row in df.iterrows():
        # Convert ObjectId to string for JSON serialization
        id_value = str(row.get('_id')) if row.get('_id') else str(_)
        
        # Extract conversation history if available
        conversation_history = []
        if 'message_history' in row and isinstance(row['message_history'], list):
            conversation_history = row['message_history']
        
        # Handle NaN values by replacing them with None (which becomes null in JSON)
        feedback_value = row.get('Feedback', "")
        if pd.isna(feedback_value):
            feedback_value = ""
        
        # Similarly handle other fields that might contain NaN
        email_value = row.get('Email', "Unknown")
        if pd.isna(email_value):
            email_value = "Unknown"
            
        question_value = row.get('Question', "")
        if pd.isna(question_value):
            question_value = ""
            
        answer_value = row.get('Answer', "")
        if pd.isna(answer_value):
            answer_value = ""
            
        conv_id = row.get('conversation_id', "")
        if pd.isna(conv_id):
            conv_id = ""
        
        chat = {
            "id": id_value,
            "date": row['Date'].strftime("%d-%b-%Y %H:%M") if 'Date' in row and not pd.isna(row['Date']) else "",
            "bot_type": "ChatBot",
            "username": email_value,
            "question": question_value,
            "answer": answer_value,
            "feedback": feedback_value,
            "conversation_id": conv_id,
            "conversation_history": conversation_history
        }
        chats.append(chat)
    
    # Print summary for debugging
    print(f"Final chat count after all filters: {len(chats)}")
    
    return jsonify({
        "chats": chats,
        "total_count": len(chats)
    })

# Add API endpoint for username suggestions - role-based filtering
@app.route('/api/usernames')
def get_usernames():
    # Check if user is authenticated
    if 'email' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        authenticated_email = session['email']
        user_role = session.get('role', 'member')
        collection = get_mongo_connection()
        
        if user_role == 'admin':
            # Admin can see all usernames
            unique_emails = collection.distinct('email')
            # Filter out empty values and sort
            unique_emails = [email for email in unique_emails if email and isinstance(email, str) and email.strip()]
            unique_emails.sort()
        else:
            # Member can only see their own email
            unique_emails = [authenticated_email]
        
        return jsonify({
            "usernames": unique_emails
        })
    except Exception as e:
        print(f"Error retrieving usernames from MongoDB: {e}")
        return jsonify({
            "usernames": []
        })

# Add API endpoint for feedback filters - role-based filtering
@app.route('/api/feedback_options')
def get_feedback_options():
    # Check if user is authenticated
    if 'email' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        authenticated_email = session['email']
        user_role = session.get('role', 'member')
        collection = get_mongo_connection()
        
        if user_role == 'admin':
            # Admin can see feedback from all users
            unique_feedback = collection.distinct('feedback')
        else:
            # Member can only see their own feedback options
            unique_feedback = collection.distinct('feedback', {'email': authenticated_email})
        
        # Filter out empty values and sort
        unique_feedback = [feedback for feedback in unique_feedback if feedback and isinstance(feedback, str) and feedback.strip()]
        unique_feedback.sort()
        
        return jsonify({
            "feedback_options": unique_feedback
        })
    except Exception as e:
        print(f"Error retrieving feedback options from MongoDB: {e}")
        return jsonify({
            "feedback_options": ["Good", "Bad", "Neutral"]  # Default options
        })

# Add API endpoint to get user info (useful for frontend to know user role)
@app.route('/api/user_info')
def get_user_info():
    # Check if user is authenticated
    if 'email' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    return jsonify({
        "email": session['email'],
        "role": session.get('role', 'member'),
        "is_admin": session.get('role', 'member') == 'admin'
    })
    
@app.route('/upload')
@login_required
def upload_file():
    return render_template('upload.html')



# Upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {
    'jpg', 'jpeg', 'png', 'pdf', 'doc', 'docx', 'txt', 'xlsx', 'csv', 'gif', 'zip', 'rar', 'json', 'xlsb'
}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB in bytes

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max request size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS





# testing 
# testing 
@app.route('/upload/multiple', methods=['POST'])
def handle_upload():
    """Handle file upload and forward to Azure service"""
    try:
        if 'email' not in session:
            return jsonify({
                'status': 'error',
                'message': 'Unauthorized'
            }), 401

        # Check if files are in the request
        if 'files' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No files selected'
            }), 400

        files = request.files.getlist('files')
        
        if not files or files[0].filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No files selected'
            }), 400

        uploaded_files = []
        errors = []
        total_chunks = 0

        # Azure service base endpoint
        azure_base_url = f"{BASE_URL}/upload/multipdf-file"

        for file in files:
            if file and allowed_file(file.filename):
                # Check file size
                file.seek(0, os.SEEK_END)
                file_size = file.tell()
                file.seek(0)  # Reset file pointer
                
                if file_size > MAX_FILE_SIZE:
                    errors.append(f'{file.filename}: File size exceeds 10MB limit')
                    uploaded_files.append({
                        'filename': file.filename,
                        'error': 'File size exceeds 10MB limit'
                    })
                    continue

                # Secure the filename
                filename = secure_filename(file.filename)
                
                try:
                    # Prepare file for upload to Azure
                    file.seek(0)  # Reset pointer before reading
                    file_content = file.read()
                    
                    # Determine MIME type based on filename; default to octet-stream
                    guessed_mime, _ = mimetypes.guess_type(filename)
                    content_type = guessed_mime or 'application/octet-stream'

                    # IMPORTANT: Azure expects 'file' (singular) as the parameter name
                    files_to_upload = {
                        'file': (filename, file_content, content_type)
                    }
                    
                    print(f"📤 Uploading {filename} to Azure...")
                    print(f"   URL: {azure_base_url}")
                    print(f"   File size: {file_size} bytes")
                    print(f"   Filename: {filename}")
                    
                    # Upload to Azure service
                    response = requests.post(
                        azure_base_url,
                        files=files_to_upload,
                        timeout=120
                    )
                    
                    print(f"📥 Azure response status: {response.status_code}")
                    print(f"📥 Azure response: {response.text}")
                    
                    # Try to parse JSON response
                    try:
                        azure_response = response.json()
                        print(f"📥 Azure response JSON: {azure_response}")
                    except ValueError as json_error:
                        # If response is not JSON, treat it as an error
                        print(f"❌ Azure returned non-JSON response: {response.text}")
                        errors.append(f'{filename}: Invalid response from server')
                        uploaded_files.append({
                            'filename': filename,
                            'error': f'Invalid server response (Status: {response.status_code})'
                        })
                        continue
                    
                    # Check if upload was successful
                    if response.status_code == 200:
                        # Check if Azure response indicates success
                        if azure_response.get('status') == 'ok':
                            # Extract file path and calculate chunks (you may need to adjust this)
                            file_path = azure_response.get('path', '')
                            message = azure_response.get('message', '')
                            
                            # Estimate chunks based on file size (adjust as needed)
                            # Assuming each chunk is ~1000 characters or 1KB
                            chunks = file_size // 1024 if file_size > 0 else 1
                            total_chunks += chunks
                            
                            uploaded_files.append({
                                'filename': azure_response.get('filename', filename),
                                'size': file_size,
                                'chunks': chunks,
                                'blob_path': file_path
                            })
                            print(f"✅ Successfully uploaded {filename}")
                            print(f"   Path: {file_path}")
                            print(f"   Message: {message}")
                        else:
                            # Azure returned 200 but with error status
                            error_msg = azure_response.get('message', azure_response.get('error', 'Unknown error'))
                            errors.append(f'{filename}: {error_msg}')
                            uploaded_files.append({
                                'filename': filename,
                                'error': error_msg
                            })
                            print(f"⚠️ Azure returned error for {filename}: {error_msg}")
                    else:
                        # Non-200 status code
                        error_msg = azure_response.get('message', azure_response.get('error', f'HTTP {response.status_code}'))
                        errors.append(f'{filename}: {error_msg}')
                        uploaded_files.append({
                            'filename': filename,
                            'error': f'{error_msg} (Status: {response.status_code})'
                        })
                        print(f"❌ Azure upload failed for {filename}: {error_msg}")
                        
                except requests.exceptions.Timeout:
                    error_msg = "Upload timeout - file may be too large or server is slow"
                    print(f"⏱️ Timeout error for {filename}")
                    errors.append(f'{filename}: {error_msg}')
                    uploaded_files.append({
                        'filename': filename,
                        'error': error_msg
                    })
                except requests.exceptions.ConnectionError as conn_error:
                    error_msg = f"Connection error - cannot reach Azure server"
                    print(f"🔌 Connection error for {filename}: {str(conn_error)}")
                    errors.append(f'{filename}: {error_msg}')
                    uploaded_files.append({
                        'filename': filename,
                        'error': error_msg
                    })
                except requests.exceptions.RequestException as req_error:
                    error_msg = f"Network error - {str(req_error)}"
                    print(f"🌐 Network error for {filename}: {str(req_error)}")
                    errors.append(f'{filename}: {error_msg}')
                    uploaded_files.append({
                        'filename': filename,
                        'error': error_msg
                    })
                except Exception as upload_error:
                    error_msg = f"Upload error - {str(upload_error)}"
                    print(f"❌ Unexpected error for {filename}: {str(upload_error)}")
                    import traceback
                    traceback.print_exc()
                    errors.append(f'{filename}: {error_msg}')
                    uploaded_files.append({
                        'filename': filename,
                        'error': error_msg
                    })
            else:
                errors.append(f"{file.filename}: Invalid file type")
                uploaded_files.append({
                    'filename': file.filename,
                    'error': 'Invalid file type'
                })

        # Determine overall status
        successful_uploads = [f for f in uploaded_files if 'error' not in f]
        failed_uploads = [f for f in uploaded_files if 'error' in f]
        
        print(f"\n📊 Upload Summary:")
        print(f"   Total files: {len(uploaded_files)}")
        print(f"   Successful: {len(successful_uploads)}")
        print(f"   Failed: {len(failed_uploads)}")
        print(f"   Total chunks: {total_chunks}")

        # Return response matching the frontend expectations
        if len(successful_uploads) > 0:
            return jsonify({
                'status': 'ok',
                'message': f'Successfully uploaded {len(successful_uploads)} of {len(uploaded_files)} file(s)',
                'total_files': len(uploaded_files),
                'successful_uploads': len(successful_uploads),
                'failed_uploads': len(failed_uploads),
                'total_chunks': total_chunks,
                'uploaded_files': uploaded_files,
                'index': 'gptindex',
                'folder_path': 'content',
                'errors': errors if errors else None
            }), 200
        else:
            # All uploads failed
            return jsonify({
                'status': 'error',
                'message': 'All file uploads failed',
                'total_files': len(uploaded_files),
                'uploaded_files': uploaded_files,
                'errors': errors
            }), 400

    except Exception as e:
        print(f"❌ Critical error in upload handler: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500
# ingestion
@app.route('/ingest/files', methods=['POST'])
def ingest_files():
    """Handle file ingestion after upload"""
    try:
        if 'email' not in session:
            return jsonify({
                'status': 'error',
                'message': 'Unauthorized'
            }), 401

        # Get data from request body
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
        
        # Extract file paths
        file_paths = data.get('file_paths', [])
        
        if not file_paths or len(file_paths) == 0:
            return jsonify({
                'status': 'error',
                'message': 'No file paths provided'
            }), 400
        
        print(f"🔄 Starting ingestion for {len(file_paths)} file(s)")
        print(f"   File paths: {file_paths}")
        
        # Azure ingestion endpoint
        ingestion_url = f"{BASE_URL}/ingest/multipdf-file"
        
        ingestion_results = []
        errors = []
        
        # Send each file path for ingestion
        for file_path in file_paths:
            try:
                print(f"📥 Ingesting file: {file_path}")
                
                # Prepare the request body
                ingestion_payload = {
                    "path": file_path
                }
                
                # Send to Azure ingestion service
                response = requests.post(
                    ingestion_url,
                    json=ingestion_payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=180  # 3 minutes timeout for ingestion
                )
                
                print(f"📥 Ingestion response status: {response.status_code}")
                print(f"📥 Ingestion response: {response.text}")
                
                # Parse response
                try:
                    ingestion_response = response.json()
                except ValueError:
                    print(f"❌ Non-JSON response from ingestion service")
                    errors.append(f'{file_path}: Invalid response from ingestion service')
                    ingestion_results.append({
                        'file_path': file_path,
                        'status': 'error',
                        'error': 'Invalid response from ingestion service'
                    })
                    continue
                
                # Check if ingestion was successful
                if response.status_code == 200:
                    if ingestion_response.get('status') == 'ok' or ingestion_response.get('success') == True:
                        ingestion_results.append({
                            'file_path': file_path,
                            'status': 'success',
                            'message': ingestion_response.get('message', 'File ingested successfully'),
                            'data': ingestion_response
                        })
                        print(f"✅ Successfully ingested: {file_path}")
                    else:
                        error_msg = ingestion_response.get('message', ingestion_response.get('error', 'Unknown error'))
                        errors.append(f'{file_path}: {error_msg}')
                        ingestion_results.append({
                            'file_path': file_path,
                            'status': 'error',
                            'error': error_msg
                        })
                        print(f"⚠️ Ingestion error for {file_path}: {error_msg}")
                else:
                    error_msg = ingestion_response.get('message', ingestion_response.get('error', f'HTTP {response.status_code}'))
                    errors.append(f'{file_path}: {error_msg}')
                    ingestion_results.append({
                        'file_path': file_path,
                        'status': 'error',
                        'error': error_msg
                    })
                    print(f"❌ Ingestion failed for {file_path}: {error_msg}")
                    
            except requests.exceptions.Timeout:
                error_msg = "Ingestion timeout"
                print(f"⏱️ Timeout error for {file_path}")
                errors.append(f'{file_path}: {error_msg}')
                ingestion_results.append({
                    'file_path': file_path,
                    'status': 'error',
                    'error': error_msg
                })
            except requests.exceptions.RequestException as req_error:
                error_msg = f"Network error - {str(req_error)}"
                print(f"🌐 Network error for {file_path}: {str(req_error)}")
                errors.append(f'{file_path}: {error_msg}')
                ingestion_results.append({
                    'file_path': file_path,
                    'status': 'error',
                    'error': error_msg
                })
            except Exception as e:
                error_msg = f"Ingestion error - {str(e)}"
                print(f"❌ Error for {file_path}: {str(e)}")
                errors.append(f'{file_path}: {error_msg}')
                ingestion_results.append({
                    'file_path': file_path,
                    'status': 'error',
                    'error': error_msg
                })
        
        # Count successful and failed ingestions
        successful_ingestions = [r for r in ingestion_results if r.get('status') == 'success']
        failed_ingestions = [r for r in ingestion_results if r.get('status') == 'error']
        
        print(f"\n📊 Ingestion Summary:")
        print(f"   Total files: {len(ingestion_results)}")
        print(f"   Successful: {len(successful_ingestions)}")
        print(f"   Failed: {len(failed_ingestions)}")
        
        # Return response
        if len(successful_ingestions) > 0:
            return jsonify({
                'status': 'ok',
                'message': f'Successfully ingested {len(successful_ingestions)} of {len(ingestion_results)} file(s)',
                'total_files': len(ingestion_results),
                'successful_ingestions': len(successful_ingestions),
                'failed_ingestions': len(failed_ingestions),
                'ingestion_results': ingestion_results,
                'errors': errors if errors else None
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': 'All file ingestions failed',
                'total_files': len(ingestion_results),
                'ingestion_results': ingestion_results,
                'errors': errors
            }), 400
            
    except Exception as e:
        print(f"❌ Critical error in ingestion handler: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500



@app.route('/api/get-files', methods=['GET'])
def get_files():
    """
    Proxy endpoint to fetch files from backend service
    Returns files grouped by email/ID, including ungrouped files under "Ungrouped" category
    Now includes file metadata (size, dates, content type)
    
    Query Parameters:
    - sort_by: 'name', 'size', 'date', 'type' (default: 'date')
    - sort_order: 'asc', 'desc' (default: 'desc')
    - email_filter: filter by specific email/ID
    """
    try:
        # Get query parameters
        sort_by = request.args.get('sort_by', 'date')
        sort_order = request.args.get('sort_order', 'desc')
        email_filter = request.args.get('email_filter', None)
        
        # Backend service URL
        backend_url = f'{BASE_URL}/list_content_files'
        
        # Call the backend service with increased timeout
        try:
            response = requests.get(backend_url, timeout=30)
        except requests.exceptions.Timeout:
            return jsonify({
                'status': 'error',
                'message': 'Backend service request timed out. Please try again later.'
            }), 504
        except requests.exceptions.ConnectionError:
            return jsonify({
                'status': 'error',
                'message': 'Could not connect to backend service. Please check your connection.'
            }), 503
        except requests.exceptions.RequestException as e:
            return jsonify({
                'status': 'error',
                'message': f'Error connecting to backend service: {str(e)}'
            }), 500
        
        # Check if request was successful
        if response.status_code != 200:
            return jsonify({
                'status': 'error',
                'message': f'Backend service returned status code: {response.status_code}'
            }), response.status_code
        
        # Get the data from backend
        backend_data = response.json()
        
        # Process and group files by email
        files_by_email = {}
        ungrouped_files = []  # Files without email/ID prefix
        total_size = 0  # Track total size of all files
        
        if 'files' in backend_data and isinstance(backend_data['files'], list):
            for file_item in backend_data['files']:
                # Handle both old format (strings) and new format (objects with metadata)
                if isinstance(file_item, str):
                    # Old format: just file path as string
                    file_path = file_item
                    file_metadata = {
                        'fileName': file_path.split('/')[-1],
                        'fullPath': file_path,
                        'size': None,
                        'sizeFormatted': 'Unknown',
                        'createdOn': None,
                        'lastModified': None,
                        'contentType': 'Unknown',
                        'etag': None
                    }
                else:
                    # New format: object with metadata
                    file_path = file_item.get('name', '')
                    file_size = file_item.get('size', 0)
                    
                    file_metadata = {
                        'fileName': file_path.split('/')[-1],
                        'fullPath': file_path,
                        'size': file_size,
                        'sizeFormatted': format_file_size(file_size),
                        'createdOn': file_item.get('created_on'),
                        'lastModified': file_item.get('last_modified'),
                        'contentType': file_item.get('content_type', 'Unknown'),
                        'etag': file_item.get('etag')
                    }
                    
                    # Add to total size
                    if file_size:
                        total_size += file_size
                
                # Check if file path contains "/"
                if '/' in file_path:
                    parts = file_path.split('/', 1)  # Split only on first "/"
                    email = parts[0]  # e.g., "awais@gmail.com"
                    file_name = parts[1]  # e.g., "DLIMS.pdf"
                    
                    # Skip if email filter is specified and doesn't match
                    if email_filter and email != email_filter:
                        continue
                    
                    if email not in files_by_email:
                        files_by_email[email] = {
                            'files': [],
                            'totalSize': 0,
                            'fileCount': 0
                        }
                    
                    # Update fileName to be just the file name without email prefix
                    file_metadata['fileName'] = file_name
                    files_by_email[email]['files'].append(file_metadata)
                    files_by_email[email]['fileCount'] += 1
                    if file_metadata['size']:
                        files_by_email[email]['totalSize'] += file_metadata['size']
                else:
                    # File without email/ID prefix
                    if not email_filter or email_filter == 'Ungrouped':
                        ungrouped_files.append(file_metadata)
        
        # Sort files within each email group
        for email in files_by_email:
            files_by_email[email]['files'] = sort_files(
                files_by_email[email]['files'], 
                sort_by, 
                sort_order
            )
            # Add formatted total size
            files_by_email[email]['totalSizeFormatted'] = format_file_size(
                files_by_email[email]['totalSize']
            )
        
        # Sort ungrouped files
        ungrouped_files = sort_files(ungrouped_files, sort_by, sort_order)
        
        # Add ungrouped files to files_by_email under "Ungrouped" category
        if ungrouped_files:
            ungrouped_total_size = sum(f.get('size', 0) for f in ungrouped_files if f.get('size'))
            files_by_email['Ungrouped'] = {
                'files': ungrouped_files,
                'totalSize': ungrouped_total_size,
                'totalSizeFormatted': format_file_size(ungrouped_total_size),
                'fileCount': len(ungrouped_files)
            }
        
        # Return processed data
        return jsonify({
            'status': 'ok',
            'total_emails': len(files_by_email) - (1 if ungrouped_files else 0),
            'total_files': len(backend_data.get('files', [])),
            'total_size': total_size,
            'total_size_formatted': format_file_size(total_size),
            'files_by_email': files_by_email,
            'ungrouped_count': len(ungrouped_files),
            'sort_by': sort_by,
            'sort_order': sort_order
        }), 200
        
    except Exception as e:
        print(f"Error in get_files endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }), 500


def format_file_size(size_bytes):
    """
    Helper function to format file size in human-readable format
    """
    if size_bytes is None or size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def sort_files(files_list, sort_by='date', sort_order='desc'):
    """
    Helper function to sort files based on specified criteria
    
    Args:
        files_list: List of file dictionaries
        sort_by: 'name', 'size', 'date', 'type'
        sort_order: 'asc' or 'desc'
    
    Returns:
        Sorted list of files
    """
    reverse = (sort_order == 'desc')
    
    if sort_by == 'name':
        return sorted(files_list, key=lambda x: x.get('fileName', '').lower(), reverse=reverse)
    elif sort_by == 'size':
        return sorted(files_list, key=lambda x: x.get('size') or 0, reverse=reverse)
    elif sort_by == 'date':
        return sorted(
            files_list, 
            key=lambda x: x.get('lastModified') or x.get('createdOn') or '', 
            reverse=reverse
        )
    elif sort_by == 'type':
        return sorted(files_list, key=lambda x: x.get('contentType', '').lower(), reverse=reverse)
    else:
        # Default: sort by date
        return sorted(
            files_list, 
            key=lambda x: x.get('lastModified') or x.get('createdOn') or '', 
            reverse=reverse
        )



# delete the file from the bot
@app.route('/api/delete_file', methods=['POST'])
def delete_file():
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({'error': 'file_path is required'}), 400
        
        # Prepare payload for backend service
        payload = {
            "file_path": file_path
        }
        
        # Call backend service with DELETE method
        backend_url = f"{BASE_URL}/delete_content_file"
        response = requests.delete(backend_url, json=payload)
        
        if response.status_code == 200:
            return jsonify({'message': 'File deleted successfully'}), 200
        else:
            return jsonify({'error': 'Failed to delete file from backend', 'details': response.text}), response.status_code
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/content/<path:file_path>', methods=['GET'])
def proxy_content_preview(file_path):
    """
    Proxy endpoint to fetch content from backend and inject CSS for white background
    This ensures we can style the content even if it's cross-origin
    """
    try:
        if 'email' not in session:
            next_url = request.full_path if request.method in ('GET', 'HEAD') else ''
            if not is_safe_url(next_url):
                next_url = ''
            return redirect(url_for('login', next=next_url or None))

        # Get theme parameter (default to 'light')
        theme = request.args.get('theme', 'light')
        
        # Construct backend URL
        backend_url = f"{BASE_URL}/content/{file_path}"
        if theme:
            separator = '&' if '?' in backend_url else '?'
            backend_url = f"{backend_url}{separator}theme={theme}"
        
        # Fetch content from backend
        response = requests.get(backend_url, timeout=30)
        content_type = response.headers.get('Content-Type', '')
        excluded_headers = {'content-encoding', 'transfer-encoding', 'connection', 'content-length'}

        # Helper to copy headers to Flask response
        def apply_headers(flask_response):
            for header, value in response.headers.items():
                header_lower = header.lower()
                if header_lower in excluded_headers:
                    continue
                if header_lower == 'content-type':
                    flask_response.headers['Content-Type'] = value
                else:
                    flask_response.headers[header] = value
            return flask_response

        # If the response is not successful, proxy it as-is (no CSS injection)
        if response.status_code != 200:
            flask_response = Response(response.content, status=response.status_code)
            return apply_headers(flask_response)
        
        # If it's HTML/text content, inject CSS for white background
        if 'text/html' in content_type or content_type.startswith('text/'):
            content = response.text
            css_injection = """
<style id="preview-override-styles">
  html, body {
    background-color: #ffffff !important;
    color: #000000 !important;
  }
  body * {
    background-color: transparent !important;
  }
  pre, code {
    background-color: #f8f9fa !important;
    color: #000000 !important;
    border: 1px solid #e9ecef !important;
    padding: 12px !important;
    border-radius: 4px !important;
  }
  .json-viewer, .json-container, [class*="json"], [id*="json"] {
    background-color: #ffffff !important;
    color: #000000 !important;
  }
  div, section, article, main {
    background-color: transparent !important;
  }
  /* Override any dark theme classes */
  [class*="dark"], [class*="black"], [class*="night"] {
    background-color: #ffffff !important;
    color: #000000 !important;
  }
</style>
"""

            if '</head>' in content:
                content = content.replace('</head>', css_injection + '</head>')
            elif '<body' in content:
                content = content.replace('<body', css_injection + '<body')
            else:
                content = css_injection + content

            flask_response = Response(content, status=200)
            return apply_headers(flask_response)

        # For binary or other content types (e.g., PDFs), proxy the content as-is
        flask_response = Response(response.content, status=200)
        return apply_headers(flask_response)
        
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Backend service request timed out'}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({'error': 'Could not connect to backend service'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)), debug=True)

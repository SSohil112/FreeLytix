from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask import send_file
from flask_sqlalchemy import SQLAlchemy
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import numpy as np
from dotenv import load_dotenv

load_dotenv()
# Flask app initialization
app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fiverr_analytics.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    profile_picture = db.Column(db.String(120), default='default.jpg')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # verification flag
    is_verified = db.Column(db.Boolean, default=False)

    # User settings
    dark_mode = db.Column(db.Boolean, default=False)
    primary_color = db.Column(db.String(7), default='#1dbf73')
    card_layout = db.Column(db.String(10), default='grid')
    date_range = db.Column(db.String(20), default='month')
    currency = db.Column(db.String(5), default='₹')
    language = db.Column(db.String(5), default='en')
    email_alerts = db.Column(db.Boolean, default=True)
    daily_summary = db.Column(db.Boolean, default=False)
    weekly_summary = db.Column(db.Boolean, default=True)
    show_earnings = db.Column(db.Boolean, default=True)
    show_ratings = db.Column(db.Boolean, default=True)
    show_orders = db.Column(db.Boolean, default=True)
    show_messages = db.Column(db.Boolean, default=True)
    default_page = db.Column(db.String(20), default='home')
    export_format = db.Column(db.String(10), default='excel')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


# Create database tables
with app.app_context():
    db.create_all()

# Data loading and plot generation functions
DATA_PATH = os.path.join("data", "fiver_clean.csv")
# For demo purposes, let's create a sample dataset if the file doesn't exist
if not os.path.exists(DATA_PATH):
    # Create sample data
    np.random.seed(42)
    sample_size = 1000
    
    categories = ['Graphic Design', 'Digital Marketing', 'Writing', 'Video Editing', 'Programming']
    genders = ['Male', 'Female']
    languages = ['English', 'Spanish', 'French', 'German', 'Hindi']
    
    sample_data = {
        'Category': np.random.choice(categories, sample_size),
        'Rating': np.random.uniform(3.5, 5.0, sample_size).round(1),
        'Total_Earning': np.random.exponential(500, sample_size).round(2),
        'Gender': np.random.choice(genders, sample_size),
        'Language': np.random.choice(languages, sample_size),
        'Price': np.random.uniform(5, 500, sample_size).round(2),
        'Orders_Completed': np.random.poisson(25, sample_size),
        'Response_Time': np.random.exponential(5, sample_size).round(1)
    }
    
    df = pd.DataFrame(sample_data)
    os.makedirs("data", exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
else:
    df = pd.read_csv(DATA_PATH)

PLOT_FOLDER = "static"
os.makedirs(PLOT_FOLDER, exist_ok=True)

def generate_plots(df):
    """Generate all plots and save in static folder"""
    # Clear existing plots
    for file in os.listdir(PLOT_FOLDER):
        if file.endswith('.png'):
            os.remove(os.path.join(PLOT_FOLDER, file))
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    
        # ===============================
    # 1. Pairplot
    # ===============================
    sns.pairplot(df)
    plt.savefig(os.path.join(PLOT_FOLDER, "pairplot.png"), dpi=100, bbox_inches="tight")
    plt.close()

    # ===============================
    # 2. Correlation Heatmap
    # ===============================
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="Greens", fmt=".2f")
    plt.title("Correlation Heatmap", fontsize=14, color="#333333")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "correlation_heatmap.png"), dpi=100, bbox_inches="tight")
    plt.close()

    # ===============================
    # 3. Earnings by Rating Groups
    # ===============================
    plt.figure()
    rating_groups = df.groupby(pd.cut(df["Rating"], bins=[0,4,4.5,5]))["Total_Earning"].sum()
    sns.barplot(x=rating_groups.index.astype(str), y=rating_groups.values, color="#1DBF73")
    plt.title("Total Earnings by Rating Groups", fontsize=14, color="#333333")
    plt.xlabel("Rating Range")
    plt.ylabel("Total Earnings ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "earnings_by_rating.png"), dpi=100, bbox_inches="tight")
    plt.close()

    # ===============================
    # 4. Avg Price by Gender
    # ===============================
    plt.figure()
    df.groupby('Gender')['Price'].mean().plot(kind='bar', color="#7CA5B8")
    plt.title("Average Price by Gender", fontsize=14, color="#333333")
    plt.ylabel("Avg Price ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "avg_price_gender.png"), dpi=100, bbox_inches="tight")
    plt.close()

    # ===============================
    # 5. Avg Review Count by Gender
    # ===============================
    plt.figure()
    df.groupby('Gender')['Review_Count'].mean().sort_values().plot(kind='bar', color="#1DBF73")
    plt.title("Average Review Count by Gender", fontsize=14, color="#333333")
    plt.ylabel("Avg Reviews")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "avg_reviews_gender.png"), dpi=100, bbox_inches="tight")
    plt.close()

    # ===============================
    # 6. Total Earnings by Gender
    # ===============================
    plt.figure()
    df.groupby('Gender')['Total_Earning'].sum().sort_values().plot(kind='bar', color="#7CA5B8")
    plt.title("Total Earnings by Gender", fontsize=14, color="#333333")
    plt.ylabel("Total Earnings ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "total_earnings_gender.png"), dpi=100, bbox_inches="tight")
    plt.close()

    # ===============================
    # 7. Avg Rating by Level
    # ===============================
    plt.figure()
    df.groupby('Level')['Rating'].mean().sort_values().plot(kind='bar', color="#1DBF73")
    plt.title("Average Rating by Level", fontsize=14, color="#333333")
    plt.ylabel("Avg Rating")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "avg_rating_level.png"), dpi=100, bbox_inches="tight")
    plt.close()

    # ===============================
    # 8. Avg Review Count by Level
    # ===============================
    plt.figure()
    df.groupby('Level')['Review_Count'].mean().sort_values().plot(kind='bar', color="#7CA5B8")
    plt.title("Average Review Count by Level", fontsize=14, color="#333333")
    plt.ylabel("Avg Reviews")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "avg_reviews_level.png"), dpi=100, bbox_inches="tight")
    plt.close()

    # ===============================
    # 9. Total Reviews by Rating
    # ===============================
    plt.figure()
    df.groupby('Rating')['Review_Count'].sum().sort_values().plot(kind='bar', color="#1DBF73")
    plt.title("Total Reviews by Rating", fontsize=14, color="#333333")
    plt.ylabel("Total Reviews")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "reviews_by_rating.png"), dpi=100, bbox_inches="tight")
    plt.close()

    # ===============================
    # 10. Boxplot Price by Level
    # ===============================
    plt.figure(figsize=(8,6))
    sns.boxplot(x="Level", y="Price", data=df, palette="Greens")
    plt.title("Price Distribution by Level", fontsize=14, color="#333333")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "price_boxplot_level.png"), dpi=100, bbox_inches="tight")
    plt.close()

    # ===============================
    # 11. Boxplot Earnings by Gender
    # ===============================
    plt.figure(figsize=(8,6))
    sns.boxplot(x="Total_Earning", y="Gender", data=df, palette="Greens")
    plt.title("Earnings Distribution by Gender", fontsize=14, color="#333333")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "earnings_boxplot_gender.png"), dpi=100, bbox_inches="tight")
    plt.close()

    # ===============================
    # 12. Boxplot Reviews by Gender
    # ===============================
    plt.figure(figsize=(8,6))
    sns.boxplot(x="Review_Count", y="Gender", data=df, palette="Greens")
    plt.title("Review Count Distribution by Gender", fontsize=14, color="#333333")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "reviews_boxplot_gender.png"), dpi=100, bbox_inches="tight")
    plt.close()

    # ===============================
    # 13. Boxplot Price by Gender
    # ===============================
    plt.figure(figsize=(8,6))
    sns.boxplot(x="Price", y="Gender", data=df, palette="Greens")
    plt.title("Price Distribution by Gender", fontsize=14, color="#333333")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "price_boxplot_gender.png"), dpi=100, bbox_inches="tight")
    plt.close()

    # ===============================
    # 14. Rating Sum by Category
    # ===============================
    plt.figure(figsize=(10,6))
    sns.barplot(x=df.groupby("Category")["Rating"].sum().index,
                y=df.groupby("Category")["Rating"].sum().values,
                color="#1DBF73")
    plt.xticks(rotation=90)
    plt.title("Total Ratings by Category", fontsize=14, color="#333333")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "ratings_by_category.png"), dpi=100, bbox_inches="tight")
    plt.close()

    # ===============================
    # 15. Avg Rating by Category
    # ===============================
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x='Category', y='Rating', color="#7CA5B8")
    plt.xticks(rotation=90)
    plt.title("Average Rating by Category", fontsize=14, color="#333333")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "avg_rating_category.png"), dpi=100, bbox_inches="tight")
    plt.close()

    # ===============================
    # 16. Avg Earnings by Category
    # ===============================
    plt.figure(figsize=(10,6))
    df.groupby('Category')['Total_Earning'].mean().sort_values().plot(kind='bar', color="#1DBF73")
    plt.xticks(rotation=90)
    plt.title("Average Earnings by Category", fontsize=14, color="#333333")
    plt.ylabel("Avg Earnings ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "avg_earnings_category.png"), dpi=100, bbox_inches="tight")
    plt.close()

    # ===============================
    # 17. Max Earnings by Category
    # ===============================
    plt.figure(figsize=(10,6))
    df.groupby('Category')['Total_Earning'].max().sort_values().plot(kind='bar', color="#7CA5B8")
    plt.xticks(rotation=90)
    plt.title("Max Earnings by Category", fontsize=14, color="#333333")
    plt.ylabel("Max Earnings ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "max_earnings_category.png"), dpi=100, bbox_inches="tight")
    plt.close()

    # ===============================
    # 18. Total Earnings by Category
    # ===============================
    plt.figure(figsize=(10,6))
    df.groupby('Category')['Total_Earning'].sum().sort_values().plot(kind='bar', color="#1DBF73")
    plt.xticks(rotation=90)
    plt.title("Total Earnings by Category", fontsize=14, color="#333333")
    plt.ylabel("Total Earnings ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "total_earnings_category.png"), dpi=100, bbox_inches="tight")
    plt.close()

    # ===============================
    # 19. Total Reviews by Category
    # ===============================
    plt.figure(figsize=(10,6))
    df.groupby('Category')['Review_Count'].sum().sort_values().plot(kind='bar', color="#7CA5B8")
    plt.xticks(rotation=90)
    plt.title("Total Reviews by Category", fontsize=14, color="#333333")
    plt.ylabel("Total Reviews")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "total_reviews_category.png"), dpi=100, bbox_inches="tight")
    plt.close()

    # ===============================
    # 20. Avg Price by Category
    # ===============================
    plt.figure(figsize=(10,6))
    sns.barplot(x=df.groupby('Category')['Price'].mean().sort_values().index,
                y=df.groupby('Category')['Price'].mean().sort_values().values,
                color="#1DBF73")
    plt.xticks(rotation=90)
    plt.title("Average Price by Category", fontsize=14, color="#333333")
    plt.ylabel("Avg Price ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "avg_price_category.png"), dpi=100, bbox_inches="tight")
    plt.close()

    # ===============================
    # 21. Countplot Category by Gender
    # ===============================
    plt.figure(figsize=(10,6))
    sns.countplot(data=df, x='Category', hue='Gender', palette="Greens")
    plt.xticks(rotation=90)
    plt.title("Category Distribution by Gender", fontsize=14, color="#333333")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "category_gender_count.png"), dpi=100, bbox_inches="tight")
    plt.close()

    # ===============================
    # 22. Price by Category & Gender
    # ===============================
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x='Category', y='Price', hue='Gender', palette="Greens")
    plt.xticks(rotation=90)
    plt.title("Price by Category & Gender", fontsize=14, color="#333333")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "price_category_gender.png"), dpi=100, bbox_inches="tight")
    plt.close()

    # ===============================
    # 23. Price by Level & Gender
    # ===============================
    plt.figure(figsize=(8,6))
    sns.barplot(data=df, x='Level', y='Price', hue='Gender', palette="Greens")
    plt.title("Price by Level & Gender", fontsize=14, color="#333333")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "price_level_gender.png"), dpi=100, bbox_inches="tight")
    plt.close()

    # ===============================
    # 24. KDE Plot Review Count
    # ===============================
    plt.figure()
    df['Review_Count'].value_counts().plot(kind='kde', color="#1DBF73")
    plt.title("KDE Distribution of Review Counts", fontsize=14, color="#333333")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "review_count_kde.png"), dpi=100, bbox_inches="tight")
    plt.close()

    # ===============================
    # 25. Value Count of Levels
    # ===============================
    plt.figure()
    df['Level'].value_counts().plot(kind='bar', color="#7CA5B8")
    plt.title("Distribution of Levels", fontsize=14, color="#333333")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, "level_distribution.png"), dpi=100, bbox_inches="tight")
    plt.close()

    
    return len(os.listdir(PLOT_FOLDER))


from flask_mail import Mail, Message
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY")
# App config (app.py me upar set karna h)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'sohilkhan971170@gmail.com'        # apna email
app.config['MAIL_PASSWORD'] = os.getenv("MAIL_PASSWORD")           # Gmail App password
app.config['MAIL_DEFAULT_SENDER'] = ('FreeLytix', 'sohilkhan971170@gmail.com')

mail = Mail(app)
serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('register.html')

        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return render_template('register.html')

        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'error')
            return render_template('register.html')

        # Create new user (initially not verified)
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        new_user.is_verified = False

        db.session.add(new_user)
        db.session.commit()

        # Send verification email with token
        token = serializer.dumps(email, salt='email-confirm-salt')
        confirm_url = url_for('confirm_email', token=token, _external=True)

        msg = Message(
            subject="Confirm your FreeLytix account",
            recipients=[email]
        )
        msg.body = f"""Hi {username},

Welcome to FreeLytix! Please confirm your email address by clicking the link below:

{confirm_url}

If you didn't sign up, just ignore this email.

Thanks,
FreeLytix Team
"""
        try:
            mail.send(msg)
            flash('Registration successful! A confirmation email has been sent. Please verify your email to log in.', 'success')
        except Exception as e:
            db.session.delete(new_user)  # rollback if email fails
            db.session.commit()
            # If mail fails, still keep user but inform admin/dev
            flash('Registration successful, but confirmation email could not be sent. Contact support.', 'error')
            app.logger.error(f"Mail send error: {e}")

        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/confirm/<token>')
def confirm_email(token):
    try:
        email = serializer.loads(token, salt='email-confirm-salt', max_age=3600)  # 1 hour expiry
    except SignatureExpired:
        flash('The confirmation link has expired. Please request a new confirmation email.', 'error')
        return redirect(url_for('resend_confirmation'))
    except BadSignature:
        flash('Invalid confirmation link.', 'error')
        return redirect(url_for('login'))

    user = User.query.filter_by(email=email).first()
    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('register'))

    if user.is_verified:
        flash('Account already verified. Please log in.', 'info')
    else:
        user.is_verified = True
        db.session.commit()
        flash('Email verified successfully! You can now log in.', 'success')

    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            if not user.is_verified:
                flash('Please verify your email before logging in. Check your inbox (or resend verification).', 'error')
                return render_template('login.html')
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'error')

    return render_template('login.html')

@app.route('/resend-confirmation', methods=['GET', 'POST'])
def resend_confirmation():
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        if user:
            if user.is_verified:
                flash('Email already verified. Please log in.', 'info')
                return redirect(url_for('login'))
            token = serializer.dumps(email, salt='email-confirm-salt')
            confirm_url = url_for('confirm_email', token=token, _external=True)
            msg = Message(subject="Confirm your FreeLytix account",
                          recipients=[email])
            msg.body = f"Click to confirm: {confirm_url}"
            try:
                mail.send(msg)
                flash('A new confirmation email has been sent.', 'success')
            except Exception as e:
                flash('Could not send confirmation email. Contact support.', 'error')
                app.logger.error(f"Mail send error: {e}")
            return redirect(url_for('login'))
        else:
            flash('No account found with that email.', 'error')
    return render_template('resend_confirmation.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('home'))

# Profile and settings routes
@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        flash('Please log in to access this page', 'error')
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    
    if request.method == 'POST':
        # Update profile information
        user.username = request.form.get('username')
        user.email = request.form.get('email')
        
        # Update password if provided
        new_password = request.form.get('new_password')
        if new_password:
            if new_password == request.form.get('confirm_password'):
                user.set_password(new_password)
                flash('Password updated successfully', 'success')
            else:
                flash('Passwords do not match', 'error')
        
        db.session.commit()
        flash('Profile updated successfully', 'success')
        return redirect(url_for('profile'))
    
    return render_template('profile.html', user=user)

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if 'user_id' not in session:
        flash('Please log in to access this page', 'error')
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    
    if request.method == 'POST':
        # Theme & Appearance
        user.dark_mode = 'dark_mode' in request.form
        user.primary_color = request.form.get('primary_color', '#1dbf73')
        user.card_layout = request.form.get('card_layout', 'grid')
        
        # Data Preferences
        user.date_range = request.form.get('date_range', 'month')
        user.currency = request.form.get('currency', '₹')
        user.language = request.form.get('language', 'en')
        
        # Notification Preferences
        user.email_alerts = 'email_alerts' in request.form
        user.daily_summary = 'daily_summary' in request.form
        user.weekly_summary = 'weekly_summary' in request.form
        
        # Dashboard Customization
        user.show_earnings = 'show_earnings' in request.form
        user.show_ratings = 'show_ratings' in request.form
        user.show_orders = 'show_orders' in request.form
        user.show_messages = 'show_messages' in request.form
        user.default_page = request.form.get('default_page', 'home')
        user.export_format = request.form.get('export_format', 'excel')
        
        db.session.commit()
        flash('Settings saved successfully', 'success')
        return redirect(url_for('settings'))
    
    return render_template('settings.html', user=user)

# Main routes
@app.route("/")
def home():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    else:
        user = None
    
    total_users = df["Category"].nunique()
    total_earnings = df["Total_Earning"].sum()
    avg_rating = df["Rating"].mean().round(2)

    return render_template("index.html",
                           total_users=total_users,
                           total_earnings=total_earnings,
                           avg_rating=avg_rating,
                           user=user)

@app.route("/charts")
def eda():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    else:
        user = None

    # Check karo ki folder me plots already hai ya nahi
    plots = sorted([f for f in os.listdir(PLOT_FOLDER) if f.endswith('.png')])

    if not plots:  # agar koi plot file nahi hai to hi naya generate karo
        num_plots = generate_plots(df)
        plots = sorted([f for f in os.listdir(PLOT_FOLDER) if f.endswith('.png')])
    else:
        num_plots = len(plots)

    return render_template("charts.html", plots=plots, user=user, num_plots=num_plots)


@app.route("/about")
def about():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    else:
        user = None
        
    return render_template("about.html", user=user)

@app.route("/download")
def download_file():
    file_path = os.path.join("data", "fiver_clean.csv")
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
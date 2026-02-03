import io
import os
import json
import random
import time
import uuid
import datetime
import bcrypt
import jwt
import re
import statistics
import requests
from flask import Flask, request, jsonify, g, redirect
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor, Json
import difflib
from PIL import Image as PILImage
from dotenv import load_dotenv
from werkzeug.exceptions import HTTPException
load_dotenv()
load_dotenv("/home/ubuntu/backend/.env")

# --- GOOGLE VISION SETUP ---
# Ensure you have 'service_account.json' in the backend folder
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account.json"

try:
    from google.cloud import vision
except ImportError:
    print("⚠️ Google Cloud Vision library not found. Install it via pip.")
    vision = None

app = Flask(__name__)
# Enable CORS for all routes and origins, allowing Authorization headers
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
# ==================================================================================
# --- CORS HANDLING ---
# ==================================================================================
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    # Pass through HTTP errors
    if isinstance(e, HTTPException):
        return e

    print(f"❌ Server Error: {e}")
    # Return JSON instead of HTML for 500s
    return jsonify({
        "error": "Internal Server Error",
        "message": str(e)
    }), 500
# ==================================================================================

# --- CONFIGURATION ---
PORT = 5000

JWT_SECRET = os.getenv("JWT_SECRET") or "fd_analytics_secret_key_change_this"
UPLOAD_FOLDER = 'uploads'
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:3000") # URL of your React App

# --- GOOGLE OAUTH CONFIG ---
# TODO: Get these from Google Cloud Console -> APIs & Services -> Credentials
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

# 🔴 CRITICAL: This URL must match EXACTLY what is in Google Cloud Console > Authorized redirect URIs
GOOGLE_REDIRECT_URI = "http://localhost:5000/api/auth/google/callback"
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:5000")
GOOGLE_REDIRECT_URI = f"{BACKEND_URL}/api/auth/google/callback"
# ==================================================================================
# 🔧 DATABASE CONFIGURATION
# ==================================================================================

DATABASE_URL = os.getenv("DATABASE_URL")


# --- GEMINI AI SETUP ---
try:
    import google.generativeai as genai
except ImportError:
    print("⚠️ google-generativeai not found. Run: pip install google-generativeai")
    genai = None

# ==================================================================================
# 🔧 DATABASE FUNCTIONS
# ==================================================================================

def get_db_connection():
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        print("❌ DB Connection Error:", e)
        raise e

def init_db():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Users Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                email TEXT UNIQUE NOT NULL,
                password TEXT,
                username TEXT,
                bgmi_id TEXT,
                bio TEXT,
                device TEXT,
                tier TEXT,
                social_links JSONB,
                auth_provider TEXT DEFAULT 'email'
            );
        """)
        
        # Teams Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS teams (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID REFERENCES users(id),
                name TEXT,
                players JSONB
            );
        """)
        
        # Matches Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID REFERENCES users(id),
                created_at TIMESTAMP,
                map TEXT,
                placement INTEGER,
                total_kills INTEGER,
                total_damage INTEGER,
                players JSONB,
                insights JSONB
            );
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Database tables initialized.")
    except Exception as e:
        print(f"⚠️ Database initialization failed: {e}")

# --- MIDDLEWARE ---
from functools import wraps

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')
        if auth_header:
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                return jsonify({'message': 'Token is missing!'}), 401
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        try:
            data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            g.user_id = data['id']
        except Exception as e:
            return jsonify({'message': 'Token is invalid!'}), 403
        return f(*args, **kwargs)
    return decorated

# ==================================================================================
# 🧠 AI ENGINE (GEMINI VISION)
# ==================================================================================

def analyze_images_with_gemini(images, known_players):
    """
    Sends images to Gemini.
    Uses strict local variable scoping to avoid NameError.
    """
    # 1. Access Environment Variable Locally
    api_key_env = os.environ.get("GEMINI_API_KEY")
    
    # 2. Validation
    if not api_key_env:
        print("❌ Error: GEMINI_API_KEY is not found in environment variables.")
        raise Exception("Server Misconfiguration: GEMINI_API_KEY is missing in .env file.")
        
    if "YOUR_" in api_key_env:
        raise Exception("Invalid API Key: You are using the placeholder. Get a real key from aistudio.google.com")

    if not genai:
        raise Exception("Python library 'google-generativeai' is not installed.")

    # 3. Configure Gemini with the local variable
    genai.configure(api_key=api_key_env)
    
    # Use Flash Lite Latest (Cost-Effective & Available)
    model_name = 'gemini-flash-lite-latest'
    model = genai.GenerativeModel(model_name)

    # Prepare known names
    roster_names = [p['name'] for p in known_players] if len(known_players) > 0 and isinstance(known_players[0], dict) else known_players
    roster_context = ", ".join(roster_names)

    # 4. Prompt
    prompt = f"""
    You are an expert Esports Analyst parsing BGMI/PUBG Mobile screenshots.
    
    YOUR GOAL: Extract match data into strict JSON format.
    
    INPUTS:
    - Images provided may include a "Match Result" screen (showing Rank # and Map) or a "Scoreboard" screen (rows of player stats).
    
    TEAM CONTEXT (Use these names to correct OCR typos): 
    [{roster_context}]

    DATA EXTRACTION RULES:
    1. **Map**: Look for map names like Erangel, Miramar, Sanhok, Vikendi, Livik, Rondo.
    2. **Placement**: Look for big rank numbers like "#1", "#2", or "1st", "2nd".
    3. **Player Stats**: Look for a table with columns. 
       - Column headers might be "Finishes" (Kills), "Damage", "Assists", "Rescue" (Revives), "Survival".
       - **Kills**: Usually a small integer (0-30).
       - **Damage**: Usually a large integer (0-5000).
       - **Survival**: Time format. If you see "23.0m", convert it to "23:00". If "18.5m", "18:30". If "21:45", keep as is.
    
    JSON OUTPUT FORMAT (No Markdown, just raw JSON):
    {{
      "map": "String",
      "placement": Integer,
      "players": [
        {{ 
           "name": "String", 
           "kills": Integer, 
           "damage": Integer, 
           "assists": Integer, 
           "revives": Integer, 
           "survivalTime": "MM:SS" 
        }}
      ]
    }}
    """

    content_payload = [prompt]
    valid_image_count = 0
    
    for img_bytes in images:
        try:
            # Use the aliased PILImage to ensure 'open' exists and no name conflict occurs
            img = PILImage.open(io.BytesIO(img_bytes))
            content_payload.append(img)
            valid_image_count += 1
        except Exception as e:
            print(f"❌ Skipping invalid image: {e}")

    if valid_image_count == 0:
        raise Exception("No valid images were processed. Ensure you uploaded JPG/PNG files.")

    # Retry Logic for Rate Limits (429)
    max_retries = 3
    base_delay = 2

    for attempt in range(max_retries):
        try:
            response = model.generate_content(content_payload)
            text = response.text
            
            # Clean up Markdown
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.replace("```", "")
                
            return json.loads(text.strip())

        except Exception as e:
            error_msg = str(e)
            is_rate_limit = "429" in error_msg or "Resource exhausted" in error_msg
            
            if is_rate_limit:
                if attempt < max_retries - 1:
                    sleep_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"⚠️ Rate Limit (429) on {model_name}. Retrying in {sleep_time:.2f}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(sleep_time)
                    continue
                else:
                    print(f"❌ Gemini AI Error: {error_msg}")
                    raise Exception(f"Service Busy (429): Rate limit exceeded after {max_retries} attempts.")
            
            # Handle other errors immediately
            print(f"❌ Gemini AI Error: {error_msg}")
            if "404" in error_msg:
                 # Try to list models to help debugging
                 try:
                    available = [m.name for m in genai.list_models()]
                    print(f"Available Models: {available}")
                    raise Exception(f"Model '{model_name}' not found. Check if your API Key supports this model.")
                 except:
                    raise Exception(f"Model '{model_name}' not found (404). Check your API Key permissions.")
            
            if "403" in error_msg or "PERMISSION_DENIED" in error_msg:
                 raise Exception("Permission Denied: Ensure your API Key is valid.")
            
            # If unexpected error, just return fallback
            return {
                "map": "Processing Failed",
                "placement": 0,
                "players": []
            }
    
    return {
        "map": "Processing Failed",
        "placement": 0,
        "players": []
    }

# ==================================================================================
# 🌐 API ROUTES
# ==================================================================================

@app.route('/')
def home():
    # Helper to check config status
    key_status = "Set" if os.environ.get("GEMINI_API_KEY") else "Missing"
    return jsonify({
        "status": "online", 
        "message": "FDAnalytics Backend 8.8 (Gemini Flash Lite Latest)", 
        "gemini_key": key_status
    }), 200

# --- USER ROUTES ---
@app.route('/api/me', methods=['GET'])
@token_required
def get_current_user():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, email, username, bgmi_id, bio, device, tier, social_links FROM users WHERE id = %s", (g.user_id,))
    user = cur.fetchone()
    cur.close()
    conn.close()
    if not user: return jsonify({'error': 'User not found'}), 404
    return jsonify(user)

@app.route('/api/me', methods=['PUT'])
@token_required
def update_profile():
    data = request.get_json()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("UPDATE users SET username=%s, bgmi_id=%s, bio=%s, device=%s, tier=%s, social_links=%s WHERE id=%s", 
               (data.get('username'), data.get('bgmi_id'), data.get('bio'), data.get('device'), data.get('tier'), Json(data.get('social_links', {})), g.user_id))
    conn.commit()
    cur.close()
    conn.close()
    return jsonify({'success': True})

# --- AUTH ROUTES ---
@app.route('/api/auth/google/url', methods=['GET'])
def get_google_auth_url():
    if "YOUR_" in GOOGLE_CLIENT_ID: return jsonify({"error": "Config missing"}), 500
    params = { "client_id": GOOGLE_CLIENT_ID, "redirect_uri": GOOGLE_REDIRECT_URI, "response_type": "code", "scope": "openid email profile", "access_type": "offline", "prompt": "select_account" }
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + "&".join([f"{k}={v}" for k, v in params.items()])
    return jsonify({"url": url})

@app.route('/api/auth/google/callback', methods=['GET'])
def google_auth_callback():
    code = request.args.get("code")
    if not code: return redirect(f"{FRONTEND_URL}/#/login?error=Google_Auth_Failed")
    try:
        r = requests.post("https://oauth2.googleapis.com/token", data={"code": code, "client_id": GOOGLE_CLIENT_ID, "client_secret": GOOGLE_CLIENT_SECRET, "redirect_uri": GOOGLE_REDIRECT_URI, "grant_type": "authorization_code"})
        if r.status_code != 200: return redirect(f"{FRONTEND_URL}/#/login?error=Google_Token_Failed")
        access_token = r.json().get("access_token")
        
        user_info = requests.get("https://www.googleapis.com/oauth2/v3/userinfo", headers={"Authorization": f"Bearer {access_token}"}).json()
        email = user_info.get("email")
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        setup = False
        
        if not user:
            uname = user_info.get("name").replace(" ", "").lower() + str(uuid.uuid4().hex[:4])
            cur.execute("INSERT INTO users (email, username, auth_provider) VALUES (%s, %s, 'google') RETURNING id", (email, uname))
            uid = cur.fetchone()['id']
            conn.commit()
            user = {'id': uid, 'email': email}
            setup = True
        else:
            cur.execute("SELECT id FROM teams WHERE user_id = %s", (user['id'],))
            if not cur.fetchone(): setup = True
            
        token = jwt.encode({'id': user['id'], 'email': email, 'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)}, JWT_SECRET)
        cur.close()
        conn.close()
        return redirect(f"{FRONTEND_URL}/#/login?token={token}&setup={str(setup).lower()}")
    except Exception as e:
        print(e)
        return redirect(f"{FRONTEND_URL}/#/login?error=Auth_Error")

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    uname = email.split('@')[0]
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO users (email, password, username) VALUES (%s, %s, %s) RETURNING id", (email, hashed, uname))
        uid = cur.fetchone()['id']
        conn.commit()
        cur.close()
        conn.close()
        token = jwt.encode({'id': uid, 'email': email, 'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)}, JWT_SECRET)
        return jsonify({'user': {'id': str(uid), 'email': email}, 'token': token, 'setupRequired': True})
    except: return jsonify({'error': 'Email exists'}), 400

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cur.fetchone()
    cur.close()
    conn.close()
    if user and user['auth_provider'] != 'google' and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM teams WHERE user_id = %s", (user['id'],))
        team = cur.fetchone()
        cur.close()
        conn.close()
        token = jwt.encode({'id': user['id'], 'email': email, 'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)}, JWT_SECRET)
        return jsonify({'user': {'id': str(user['id']), 'email': email}, 'token': token, 'setupRequired': not team})
    return jsonify({'error': 'Invalid credentials'}), 400

# --- TEAM ---
@app.route('/api/team', methods=['GET', 'POST'])
@token_required
def handle_team():
    conn = get_db_connection()
    cur = conn.cursor()
    if request.method == 'GET':
        cur.execute("SELECT * FROM teams WHERE user_id = %s", (g.user_id,))
        team = cur.fetchone()
        cur.close()
        conn.close()
        return jsonify(team)
    if request.method == 'POST':
        d = request.get_json()
        cur.execute("SELECT id FROM teams WHERE user_id = %s", (g.user_id,))
        existing = cur.fetchone()
        if existing:
            cur.execute("UPDATE teams SET name=%s, players=%s WHERE user_id=%s", (d['name'], Json(d['players']), g.user_id))
            tid = existing['id']
        else:
            tid = str(uuid.uuid4())
            cur.execute("INSERT INTO teams (id, user_id, name, players) VALUES (%s, %s, %s, %s)", (tid, g.user_id, d['name'], Json(d['players'])))
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({'id': tid})

# --- MATCHES ---
@app.route('/api/matches', methods=['GET', 'POST'])
@token_required
def handle_matches():
    conn = get_db_connection()
    cur = conn.cursor()
    if request.method == 'GET':
        cur.execute("SELECT * FROM matches WHERE user_id = %s ORDER BY created_at DESC", (g.user_id,))
        matches = cur.fetchall()
        cur.close()
        conn.close()
        res = []
        for m in matches:
            res.append({'id': m['id'], 'createdAt': m['created_at'], 'map': m['map'], 'placement': m['placement'], 'totalTeamKills': m['total_kills'], 'totalTeamDamage': m['total_damage'], 'players': m['players'], 'insights': m['insights']})
        return jsonify(res)
    if request.method == 'POST':
        d = request.get_json()
        mid = str(uuid.uuid4())
        tk = sum(p.get('kills', 0) for p in d['players'])
        td = sum(p.get('damage', 0) for p in d['players'])
        cur.execute("INSERT INTO matches (id, user_id, created_at, map, placement, total_kills, total_damage, players, insights) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)", (mid, g.user_id, datetime.datetime.now().isoformat(), d['map'], d['placement'], tk, td, Json(d['players']), Json(d['insights'])))
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({'success': True})

@app.route('/api/matches/<match_id>', methods=['DELETE'])
@token_required
def delete_match(match_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM matches WHERE id = %s AND user_id = %s", (match_id, g.user_id))
    conn.commit()
    cur.close()
    conn.close()
    return jsonify({'success': True})

# --- NEW: AI COACH (SQUAD ANALYSIS) ---
@app.route('/api/coach/analyze', methods=['POST'])
@token_required
def analyze_squad():
    matches = request.get_json().get('matches', [])
    if not matches or len(matches) == 0:
         return jsonify({'suggestions': ["Upload more matches to enable AI coaching."]})
    
    api_key_env = os.environ.get("GEMINI_API_KEY")
    if not api_key_env:
        return jsonify({'suggestions': ["AI key missing. Check backend."]}), 500
        
    try:
        genai.configure(api_key=api_key_env)
        # Use Flash Lite Latest here as well
        model = genai.GenerativeModel('gemini-flash-lite-latest')
        
        # Format match history for the AI
        summary_text = "Here is the recent match history for a BGMI esports squad:\n"
        for i, m in enumerate(matches[:10]): # Last 10 matches
             summary_text += f"Match {i+1}: Map {m.get('map')}, Placed #{m.get('placement')}, Team Kills: {m.get('totalTeamKills')}, Team Dmg: {m.get('totalTeamDamage')}\n"
             
        prompt = f"""
        {summary_text}
        
        Act as a strict and professional Esports Coach for BGMI (PUBG Mobile).
        Based on these stats, provide 3 specific, high-level tactical recommendations for the WHOLE SQUAD to improve their game.
        Focus on: Rotations, Team Synergy, and Aggression Control.
        
        Output format: Return ONLY a raw JSON array of strings. Example: ["Tip 1", "Tip 2", "Tip 3"]
        Do not use Markdown formatting.
        """
        
        response = model.generate_content(prompt)
        text = response.text
        if "```json" in text: text = text.split("```json")[1].split("```")[0]
        elif "```" in text: text = text.replace("```", "")
        
        suggestions = json.loads(text.strip())
        return jsonify({'suggestions': suggestions})
        
    except Exception as e:
        print("AI Coach Error:", e)
        return jsonify({'suggestions': ["Focus on team coordination.", "Review your drop spots.", "Improve communication during rotations."]})


# --- PROCESS IMAGE (GEMINI POWERED) ---
@app.route('/api/process-match', methods=['POST'])
@token_required
def process_match():
    if 'images' not in request.files: return jsonify({'error': 'No images uploaded'}), 400
    files = request.files.getlist('images')
    
    # Get user's team for context
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT players FROM teams WHERE user_id = %s", (g.user_id,))
    team = cur.fetchone()
    cur.close()
    conn.close()
    
    known = []
    if team and team['players']:
        known = team['players'] if isinstance(team['players'], list) else json.loads(team['players'])

    # Read file bytes
    image_bytes_list = []
    for f in files:
        image_bytes_list.append(f.read())

    # Call Gemini
    try:
        data = analyze_images_with_gemini(image_bytes_list, known)
        if not data:
            return jsonify({'error': 'AI could not process the image. Try a clearer screenshot.'}), 400
        return jsonify(data)
    except Exception as e:
        print("Backend Error:", e)
        # Send the actual error message to frontend
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db() 
    print("🚀 Server running on http://localhost:5000")
    print(f"🔑 Gemini Key Status: {'Loaded' if os.environ.get('GEMINI_API_KEY') else 'MISSING (Check .env)'}")
    app.run(port=PORT, debug=True)
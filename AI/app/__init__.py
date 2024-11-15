from flask import Flask
from dotenv import load_dotenv
from .routes.model_routes import model_bp
from flask_cors import CORS

# Load environment variables
load_dotenv()

def create_app():
    app = Flask(__name__)
    app.debug = True 
    app.register_blueprint(model_bp, url_prefix='/model')
    CORS(app)


    return app

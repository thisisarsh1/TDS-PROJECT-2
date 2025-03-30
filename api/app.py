import subprocess
import os
import textwrap
import logging
from flask import Flask, request, jsonify
from utils.question_matching import find_similar_question
from utils.file_process import unzip_folder
from utils.function_definations_llm import function_definitions_objects_llm
from utils.openai_api import extract_parameters
from utils.solution_functions import functions_dict

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Create the temporary uploads directory if it doesn't exist
TMP_DIR = "tmp_uploads"
os.makedirs(TMP_DIR, exist_ok=True)

app = Flask(__name__)

# Load secret password from environment variables
SECRET_PASSWORD = os.getenv("SECRET_PASSWORD")


@app.route("/", methods=["POST"])
def process_file():
    """Handles incoming POST requests with a question and an optional file."""
    try:
        question = request.form.get("question")
        file = request.files.get("file")  # Get the uploaded file (optional)

        if not question:
            return jsonify({"error": "Missing 'question' field"}), 400

        # Find the matched function
        matched_function, matched_description = find_similar_question(question)
        logging.info(f"Matched function: {matched_function}")

        # Handle file uploads if present
        tmp_dir = TMP_DIR  # Default temp directory
        file_names = []
        if file:
            tmp_dir, file_names = unzip_folder(file)

        # Extract parameters for the function
        function_definitions = function_definitions_objects_llm.get(matched_function, {})
        parameters = extract_parameters(str(question), function_definitions)

        # Ensure parameters are iterable (avoiding 'NoneType' errors)
        if parameters is None:
            parameters = []

        # Fetch the corresponding solution function
        solution_function = functions_dict.get(
            matched_function, lambda *args: "No matching function found"
        )

        # Call the solution function
        if file:
            answer = solution_function(file, *parameters)
        else:
            answer = solution_function(*parameters)

        # Format the output for readability
        formatted_answer = textwrap.dedent(str(answer)).strip()

        return jsonify({"answer": formatted_answer})

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/redeploy", methods=["GET"])
def redeploy():
    """Triggers redeployment using a shell script, if the correct password is provided."""
    password = request.args.get("password")

    if password != SECRET_PASSWORD:
        return "Unauthorized", 403

    try:
        subprocess.run(["bash", "../redeploy.sh"], check=True)
        return "Redeployment triggered!", 200
    except subprocess.CalledProcessError as e:
        logging.error(f"Redeployment failed: {e}")
        return jsonify({"error": f"Redeployment failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)

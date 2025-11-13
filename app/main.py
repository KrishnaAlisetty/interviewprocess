import openai
import logging
import os


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = "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"  # <-- placeholder
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

def generate_questions(common_skills, num_questions, difficulty_level):
    """
    Ask the model to write N interview questions, tuned by difficulty.
    The prompt forces numbering to start at 1 and be sequential.
    """
    prompt = f"""Based on the following common technical skills, generate {num_questions} interview questions with a difficulty level of {difficulty_level}.
                 Common Skills: {common_skills}.
                 The generated question numbers should start from 1. and must be strictly sequential (1., 2., 3., ...). Do not skip numbers. Do not start from 0."""

    logger.info("[Non-Agentic] Generating interview questions ...")
    response = openai.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=0.7,  # a bit more creative for question writing
        max_tokens=1000,
    )
    post_result_to_db(response.choices[0].text.strip().splitlines())


def post_result_to_db(lines):
    project_root = os.getcwd()
    output_folder_name = "out"

    # Construct the full path to the output directory
    output_dir = os.path.join(project_root, output_folder_name)

    # Define the filename
    file_name = "output_data.txt"
    file_path = os.path.join(output_dir, file_name)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    #url = "http://127.0.0.1:8000/users/"
    #payload = {"name": "Test", "qa": [], "jd": jd_text_for_payload}  # Example URL for testing POST requests
    for line in lines:
        answer = openai.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=line,
            temperature=0.7,
            max_tokens=50,
        ).choices[0].text.strip()

        try:
            with open(file_path, "w") as f:
                f.write(line + "\n")
                f.write(answer + "\n")
            print(f"File '{file_name}' successfully created in '{output_dir}'.")
        except IOError as e:
            print(f"Error writing to file: {e}")


if __name__ == "__main__":
    generate_questions("java, spring, springboot", "1", "medium")
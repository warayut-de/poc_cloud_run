from flask import Flask, request, jsonify
from vertexai.preview.generative_models import GenerativeModel
import json, vertexai

def extract_json_from_markdown(text):
    cleaned = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise Exception(json.dumps({
            "status": "failed",
            "status_code": 500,
            "error": f"extract_json_from_markdown: {str(e)}"
        }))

def is_valid_genai_format(data: dict) -> bool:
    try:
        if not all(k in data for k in ["response", "prompt_version", "response_text"]):
            return False
        files = data["response_text"].get("files", [])
        if not isinstance(files, list) or not files:
            return False
        first_file = files[0]
        if not all(k in first_file for k in ["filename", "contents"]):
            return False
        contents = first_file["contents"]
        if not all(k in contents for k in ["summary", "answers", "others"]):
            return False
        return True
    except Exception:
        return False

class GeminiAnalyzeAPI:
    def __init__(self, project_id: str = "tqm-ai-sandbox", location: str = "us-central1"):
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel("gemini-2.0-flash-001")

    def generate_and_parse_json(self, prompt):
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.0,
                    "top_p": 0.8,
                    "max_output_tokens": 8192
                }
            )
            json_data = extract_json_from_markdown(response.text)
            if not is_valid_genai_format(json_data):
                raise Exception(json.dumps({
                    "status": "failed",
                    "status_code": 500,
                    "error": "Invalid GenAI output format"
                }))
            return json_data
        except Exception as e:
            raise Exception(json.dumps({
                "status": "failed",
                "status_code": 500,
                "error": f"generate_and_parse_json: {str(e)}"
            }))

    def handle_request(self, request_json):
        if not request_json or "content" not in request_json:
            raise Exception(json.dumps({
                "status": "failed",
                "status_code": 400,
                "error": "content is required"
            }))
        content = request_json.get("content", "")

        with open("system_prompt.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()
        with open("ai_prompt.txt", "r", encoding="utf-8") as f:
            ai_prompt = f.read()

        full_prompt = f"""
#SYSTEM
{system_prompt}

#OUTPUT FORMAT
ให้คุณตอบกลับมาเป็น JSON ตามรูปแบบนี้เท่านั้น ไม่มีคำอธิบาย ไม่มีหัวข้อ ไม่มี Markdown
{ai_prompt}

#USER
วิเคราะห์ข้อมูลบทสนทนาต่อไปนี้ ให้อยู่ในรูปแบบที่กำหนด
{content}
"""
        result = self.generate_and_parse_json(full_prompt)
        return result

# ---------- เพิ่ม Flask App ----------
app = Flask(__name__)

@app.route("/", methods=["POST"])
def predict():
    try:
        request_json = request.get_json()
        api = GeminiAnalyzeAPI()
        result = api.handle_request(request_json)
        return jsonify(result)
    except Exception as e:
        # จะ return json error message
        return str(e), 500
